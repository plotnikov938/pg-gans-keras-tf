import time
import numpy as np
import os
import warnings
import random

import tensorflow as tf
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt

from adversarials import AdversarialLosses
from dataset import Buffer
from utils import mixup_func, PixelNormalization


# TODO: Remove later
initializer = 'he_normal'
wgan_gp = False


# TODO: Move to the utils.py
def plot(samples, rows, cols, title=None):
    samples = samples[:rows*cols]

    img_size = samples.shape[1]
    channels = samples.shape[-1]

    reshaped = (samples.reshape(rows, cols, img_size, img_size, channels)
                .transpose(0, 2, 1, 3, 4)
                .reshape(rows * img_size, cols * img_size, channels))

    reshaped = np.clip(reshaped, -1, 1)

    fig = plt.figure(figsize=(cols, rows))
    if reshaped.shape[-1] == 1:
        plt.imshow(np.squeeze(reshaped), cmap='Greys_r',
                   interpolation='nearest')
    else:
        plt.imshow(reshaped * 0.5 + 0.5,
                   interpolation='nearest')

    plt.title(title)
    plt.axis('off')

    return fig


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        return x


def downscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Downscale2D'):
        ksize = [1, factor, factor, 1]
        return tf.nn.avg_pool2d(x, ksize=ksize, strides=ksize, padding='VALID',
                                data_format='NHWC')  # NOTE: requires tf_config['graph_options.place_pruned_graph'] =


class MinibatchStdev(layers.Layer):
    """
    In this `MinibatchStdev` impementation minibatching is not supported and
    instead of concatenation the inputs with the stdev, adding stdev is done
    by replacing the last channel of the inputs.
    """

    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        # Subtract the mean value for each pixel across channels over the group
        inputs -= tf.reduce_mean(inputs, axis=0, keepdims=True)

        # calculate the average of the squared differences (variance)
        mean_sq_diff = tf.reduce_mean(tf.math.square(inputs), axis=0, keepdims=True)

        stdev = tf.math.sqrt(mean_sq_diff + 1e-8)

        # calculate the mean standard deviation across each pixel coord
        mean_pix = tf.reduce_mean(stdev, keepdims=True)

        # scale this up to be the size of one input feature map for each sample
        shape = tf.shape(inputs)
        output = tf.tile(mean_pix, (shape[0], shape[1], shape[2], 1))

        # replace the last channel of the inputs with the stdev by concatenation
        return tf.concat([inputs[..., :-1], output], axis=-1)


class LayerPG(layers.Layer):
    def __init__(self, layer, filters, kernel_size, padding='same', strides=(1, 1), act=None, scale=1., **kwargs):

        super(LayerPG, self).__init__(**kwargs)

        assert scale > 0  # Scale factor must be greater than zero
        self.layer = layer
        self.scale = scale
        self.kernel_size = kernel_size
        self.act = act

        self.drop = layers.Dropout(rate=0.0)

        self.layer_0 = layer(filters=filters,
                             kernel_size=kernel_size,
                             padding=padding,
                             strides=strides,
                             activation=act,
                             kernel_initializer=initializer)

        self.bn = layers.BatchNormalization()
        self.pn = PixelNormalization()

    def build(self, input_shapes):

        self.layer_0_0 = self.layer(filters=int(input_shapes[-1]),
                                    kernel_size=self.kernel_size,
                                    padding='same',
                                    activation=self.act,
                                    kernel_initializer=initializer)


    def call(self, inputs, drop_rate=0.0, batch_norm=None, pixel_norm=None, training=None, **kwargs):

        # Upscale the images
        if self.scale > 1:
            def get_scaled_dim(dim):
                return round(int(inputs.shape[dim])*self.scale)
            if wgan_gp:
                inputs = upscale2d(inputs)
            else:
                inputs = tf.image.resize_bilinear(inputs,
                                                  (get_scaled_dim(1), get_scaled_dim(2)),
                                                  align_corners=True)

        # Apply dropout here
        self.drop.rate = drop_rate

        inputs = self.drop(inputs, training=training)

        if True:
            outputs = self.layer_0_0(inputs)

            if batch_norm:
                outputs = self.bn(outputs, training=training)

            if pixel_norm:
                inputs = self.pn(outputs)

        outputs = self.layer_0(inputs)

        if batch_norm:
            outputs = self.bn(outputs, training=training)

        if pixel_norm:
            outputs = self.pn(outputs)

        # Downscale the images
        if self.scale < 1:
            def get_scaled_dim(dim):
                return round(int(outputs.shape[dim])*self.scale)
            if wgan_gp:
                outputs = downscale2d(outputs)
            else:
                outputs = tf.image.resize_bilinear(outputs,
                                                   (get_scaled_dim(1), get_scaled_dim(2)),
                                                   align_corners=True)

        return outputs


class DecoderPG(models.Model):
    def __init__(self, train_stages, channels, act=None, input_units=None, output_act='tanh',
                 drop_rate=0.0, batch_norm=None, pixel_norm=None, **kwargs):
        super(DecoderPG, self).__init__(**kwargs)

        self.train_stages = train_stages
        self.drop_rate = drop_rate
        self.batch_norm = batch_norm
        self.pixel_norm = pixel_norm
        self.act = act
        self.input_units = input_units or []
        self.output_act = output_act

        self.layers_pg, self.layers_to_rgb = [], []

        self.transition = layers.Lambda(lambda x: (x[0] - x[1]) * x[2] + x[1])
        self.resize = layers.Lambda(lambda args: tf.image.resize_bilinear(args[0], args[1].shape[1:3],
                                                                          align_corners=True))
        self.dense_list = [layers.Dense(units,
                                        activation=None,
                                        kernel_initializer=initializer) for units in
                           list(self.input_units)]

        self.concat = layers.Lambda(lambda x: tf.concat(x, axis=-1), name='concatenate')
        self.expand_dims = layers.Lambda(lambda x: x[:, None, None], name='expand_dims')

        for train_stage in self.train_stages:
            filters = train_stage['size'][-1]
            if not self.layers_pg:
                layer_pg = LayerPG(layers.Conv2DTranspose, filters, train_stage['size'][0],
                                   padding='valid', act=self.act, scale=1)
            else:
                layer_pg = LayerPG(layers.Conv2D, filters, 3,
                                   padding='same', act=self.act, scale=2)

            # Create rgb img for each output
            layer_to_rgb = layers.Conv2D(channels, 1,
                                         padding='same',
                                         activation=self.output_act,
                                         kernel_initializer=initializer)

            self.layers_pg.append(layer_pg)
            self.layers_to_rgb.append(layer_to_rgb)

    def call(self, inputs, stage=-1, alpha=1.0, training=None):
        if stage < 0:
            stage = len(self.train_stages) + stage

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if len(inputs) == 1:
            noise, labels_emb = inputs[0], None
        else:
            noise, labels_emb = inputs

        output = noise

        # Concatenate noise vector with labels embedding vector
        if labels_emb is not None:
            output = self.concat([output, labels_emb])

        for dense in self.dense_list:
            output = dense(output)

        # Prepare inputs before feeding to the convolution layer
        output = self.expand_dims(output)

        output_imgs = []
        for layer_pg, layer_to_rgb in zip(self.layers_pg[:stage + 1], self.layers_to_rgb[:stage + 1]):
            # TODO: Test
            # No batch_norm and pixel_norm for the output_img
            # output_img = layer_to_rgb(layer_pg(output,
            #                                    drop_rate=self.drop_rate,
            #                                    training=training))

            # TODO: Move args
            output = layer_pg(output,
                              drop_rate=self.drop_rate,
                              batch_norm=self.batch_norm,
                              pixel_norm=self.pixel_norm,
                              training=training)

            output_img = layer_to_rgb(output)
            output_imgs.append(output_img)

        if len(output_imgs) > 1:
            if wgan_gp:
                upscaled_imgs = upscale2d(output_imgs[-2])
            else:
                upscaled_imgs = self.resize([output_imgs[-2], output_imgs[-1]])
            return self.transition([output_imgs[-1], upscaled_imgs, alpha])
        else:
            return output_imgs[-1]


class EncoderPG(models.Model):
    def __init__(self, train_stages, latent_size, act=None, output_units=1, output_act=None,
                 drop_rate=0.0, batch_norm=None, pixel_norm=None, **kwargs):
        super(EncoderPG, self).__init__(**kwargs)

        self.train_stages = train_stages
        self.drop_rate = drop_rate
        self.batch_norm = batch_norm
        self.pixel_norm = pixel_norm
        self.act = act
        self.output_act = output_act
        self.latent_size = latent_size
        self.output_units = output_units or []

        self.concat = layers.Lambda(lambda x: tf.concat(x, axis=-1), name='concatenate')
        self.transition = layers.Lambda(lambda x: (x[0] - x[1]) * x[2] + x[1])
        self.resize = layers.Lambda(lambda args: tf.image.resize_bilinear(args[0], args[1].shape[1:3],
                                                                          align_corners=True))
        self.mb_std = MinibatchStdev()

        # Collect all layers into the list
        self.layers_pg, self.layers_from_rgb = [], []
        for train_stage in train_stages:
            filters = train_stage['size'][-1]

            layer_from_rgb = layers.Conv2D(filters=filters,
                                           kernel_size=1,
                                           padding='same',
                                           activation=act,
                                           kernel_initializer=initializer)

            self.layers_from_rgb.append(layer_from_rgb)

            layer_pg = LayerPG(layers.Conv2D, filters, 3, act=act, scale=0.5)
            self.layers_pg.append(layer_pg)

        self.tile = layers.Lambda(lambda x: tf.tile(
            x[:, None, None], [1, train_stages[0]['size'][0], train_stages[0]['size'][1], 1]), name='tile')
        self.concat = layers.Concatenate()
        self.conv_0 = layers.Conv2D(filters=train_stages[0]['size'][-1],
                                    kernel_size=3,
                                    activation=self.act,
                                    padding='same',
                                    kernel_initializer=initializer)
        self.drop = layers.Dropout(rate=self.drop_rate)
        self.conv_1 = layers.Conv2D(filters=train_stages[0]['size'][-1],
                                    kernel_size=train_stages[0]['size'][0],
                                    activation=self.act,
                                    kernel_initializer=initializer)
        self.flat = layers.Flatten()
        self.dense_list = [layers.Dense(units,
                                        activation=self.output_act,
                                        kernel_initializer=initializer) for units in list(self.output_units)]

    def call(self, inputs, stage=-1, alpha=1.0, training=None):
        if stage < 0:
            stage = len(self.train_stages) + stage

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        if len(inputs) == 1:
            input_imgs, labels_emb = inputs[0], None
        else:
            input_imgs, labels_emb = inputs

        output, img_large = self.layers_from_rgb[stage](input_imgs), input_imgs

        output = self.mb_std(output)

        flag_trans = False
        for layer_pg in self.layers_pg[:stage][::-1]:
            output = layer_pg(output,
                              drop_rate=self.drop_rate,
                              batch_norm=self.batch_norm,
                              pixel_norm=self.pixel_norm,
                              training=training)

            if near:
              downscaled_imgs = downscale2d(img_large)
            else:
              downscaled_imgs = self.resize([img_large, output])
            downscaled_from_rgb = self.layers_from_rgb[stage - 1](downscaled_imgs)
            # Smooth transition between inputs
            if not flag_trans:
                output = self.transition([output, downscaled_from_rgb, alpha])
                flag_trans = True

            img_large = downscaled_imgs
            stage -= 1

        # Concatenate output vector with labels embedding vector
        if labels_emb is not None:
            tiled = self.tile(labels_emb)
            output = self.concat([output, tiled])

        # Convolution outputs [batch, 1, 1, 1]
        output = self.drop(self.conv_0(output))
        output = self.conv_1(output)

        # Squeeze height and width dimensions
        output = self.flat(output)

        for dense in self.dense_list:
            output = dense(output)

        return output


class GAN_PG:
    def __init__(self, train_stages, channels, latent_size, labels_emb_size=None, mixup_alpha=None,
                 dis_act=None, gen_act=None, drop_rate=0.0, batch_norm=False, pixel_norm=False,
                 freeze_trained=False, dis_train_iters=1, gen_train_iters=1, buffer_size=False, buffer_epoch_depth=1, **kwargs):

        self.train_stages = train_stages
        self.channels = channels
        self.latent_size = latent_size
        self.freeze_trained = freeze_trained
        self.buffer_size = buffer_size
        self.buffer_epoch_depth = buffer_epoch_depth
        self.mixup_alpha = mixup_alpha or 0.0
        self.labels_emb_size = labels_emb_size
        self.dis_train_iters = dis_train_iters
        self.gen_train_iters = gen_train_iters

        self.alpha = tf.Variable(1.0, name='alpha', trainable=False)
        self.learning_rate = None

        self.generator = DecoderPG(self.train_stages, channels, act=gen_act,
                                   input_units=[latent_size], output_act='tanh', drop_rate=0.0,
                                   batch_norm=batch_norm, pixel_norm=pixel_norm, name='generator')
        self.discriminator = EncoderPG(self.train_stages, latent_size,
                                       act=dis_act, output_units=[1], output_act=None, drop_rate=drop_rate,
                                       batch_norm=False, pixel_norm=False, name='discriminator')

        self.get_loss = None
        self.optimizer = None
        self.buffer_store_proba = None

    def __call__(self, stage_num, training=None, tpu_strategy=None, resolver=None, config=None):
        assert self.train_stages[stage_num]  # There is no this training stage num.

        self.stage_num = stage_num
        self.training = training
        self.tpu_strategy = tpu_strategy
        self.resolver = resolver

        if self.tpu_strategy:
            self.sess = tf.Session(target=resolver.master(), config=config)
        else:
            self.sess = tf.Session(config=config)

        self.stage_shape = self.train_stages[self.stage_num]['size'][:2]

        # Pre_init the subclassed model
        gen_inputs = tf.keras.layers.Input(self.latent_size)
        dis_inputs = tf.keras.layers.Input((*self.stage_shape, self.channels))

        if self.labels_emb_size:
            gen_inputs = [gen_inputs, tf.keras.layers.Input(self.labels_emb_size)]
            dis_inputs = [dis_inputs, tf.keras.layers.Input(self.labels_emb_size)]

        self.generator(gen_inputs, training=self.training)
        self.discriminator(dis_inputs, training=self.training)

    def train_step(self, inputs):
        # TODO:
        if self.buffer_size:
            inputs, buffer_inputs = inputs

            if self.labels_emb_size:
                buffer_fake_imgs, buffer_labels_emb = buffer_inputs
            else:
                buffer_fake_imgs, buffer_labels_emb = buffer_inputs + (None,)

            noise_batch_size = tf.shape(buffer_fake_imgs)[0]

        if self.labels_emb_size:
            real_imgs, labels_emb = inputs
        else:
            real_imgs, labels_emb = inputs, None

        if not self.buffer_size:
            noise_batch_size = tf.shape(real_imgs)[0]

        latent_noise = tf.random.normal((noise_batch_size, self.latent_size), mean=0.0, stddev=1.0,
                                        name='latent_noise')
        inputs = [latent_noise, labels_emb[:noise_batch_size]] if self.labels_emb_size else latent_noise

        fake_images = self.generator(inputs, training=self.training)[self.stage_num]

        # Concatenate the new generated images with the images sampled
        # from the buffer if needed. Same goes for the labels_emb.
        if self.buffer_size and self.labels_emb_size:
            fake_images = tf.concat([fake_images, buffer_fake_imgs], axis=0)
            fake_labels_emb = tf.concat([labels_emb[:noise_batch_size], buffer_labels_emb], axis=0)
        else:
            fake_labels_emb = labels_emb

        # Apply mixup
        real_images_mixuped, fake_images_mixuped = mixup_func([real_imgs, fake_images],
                                                              [fake_images, real_imgs], self.mixup_alpha)

        # Pass real and fake images into discriminator separately
        self.real_logits = self.discriminator([real_images_mixuped, labels_emb],
                                              alpha=self.alpha, training=self.training)
        self.fake_logits = self.discriminator([fake_images_mixuped, fake_labels_emb],
                                              alpha=self.alpha, training=self.training)

        # Configure the losses here
        # `discrim_gp` here is in case you need to minimize the losses with gradient penalty
        discrim_gp = lambda x: self.discriminator([x, labels_emb], alpha=self.alpha, training=self.training)
        self.discriminator_loss, self.generator_loss = self.get_loss(self.real_logits, self.fake_logits,
                                                                     discriminator=discrim_gp,  # Takes only one arg
                                                                     lam=10,
                                                                     samples_real=real_images_mixuped,
                                                                     samples_fake=fake_images_mixuped)

        self.discriminator_loss = tf.reduce_sum(self.discriminator_loss) * (1. / self.batch_size)
        self.generator_loss = tf.reduce_sum(self.generator_loss) * (1. / self.batch_size)

        update_gen_vars = self.train_generator_op = self.optimizer. \
            minimize(self.generator_loss * self.loss_weights[0],
                     var_list=self.generator.trainable_variables)

        update_dis_vars = self.train_discriminator_op = self.optimizer. \
            minimize(self.discriminator_loss * self.loss_weights[1],
                     var_list=self.discriminator.trainable_variables)

        with tf.control_dependencies([update_gen_vars]):
            loss_gen = tf.identity(self.generator_loss)

        with tf.control_dependencies([update_dis_vars]):
            loss_dis = tf.identity(self.discriminator_loss)

        buffer_returns = fake_images[:noise_batch_size]
        buffer_returns = buffer_returns, labels_emb[:noise_batch_size] if self.labels_emb_size else buffer_returns

        return buffer_returns, loss_gen, loss_dis

    # TODO: Delete
    def predict(self,
                x,
                batch_size=None):

        batch_size = batch_size or len(x[0])

        # TODO: with emb and without
        if not batch_size:
            return self.sess.run(self.generated_images)

    def compile_model(self,
                      optimizer,
                      loss,
                      loss_weights=None):
        self.optimizer = optimizer
        if isinstance(loss, str):
            self.gan_mode = loss
            self.get_loss = AdversarialLosses(mode=self.gan_mode)
        else:
            self.gan_mode = 'custom'
            self.get_loss = lambda real_logits, fake_logits, **kwargs: loss(real_logits, fake_logits)

        # A tricky way to set up the learning rate on the fly during training with `learning_rate_scheduler`
        self.learning_rate = tf.Variable(self.optimizer._lr, name='learning_rate', trainable=False)
        self.optimizer._lr_t = self.learning_rate

        self.loss_weights = loss_weights or [1, 1]

    def _get_buffer_dataset(self, shapes):
        buffer_dtypes = (np.float32,)
        if self.labels_emb_size:
            buffer_dtypes += (tf.float32,)

        # Calculate the `buffer_store_probability`
        calls_to_fully_update_buffer = self.buffer_size // (self.batch_size // 2)
        buffer_store_calls_per_epoch = shapes[0] // self.batch_size // (self.dis_train_iters + self.gen_train_iters)
        max_buffer_store_calls = buffer_store_calls_per_epoch * self.buffer_epoch_depth
        self.buffer_store_proba = np.clip(calls_to_fully_update_buffer / max_buffer_store_calls, 0, 1)

        if self.buffer_store_proba >= 1:
            warnings.warn('The  probabylity to store in the buffer is greater than or equal to 1.' +
                          'This means that you should encrease `buffer_epoch_depth` or decrease `buffer_size`.')

        # Create the buffer
        img_initializer = lambda shape, dtype: np.random.normal(size=shape)
        labels_initializer = lambda shape, dtype: tf.keras.utils.to_categorical(
            np.random.randint(0, self.labels_emb_size, size=(shape[0], 1)), self.labels_emb_size)

        if self.labels_emb_size:
            input_shapes = (shapes[1:], (self.labels_emb_size,))
            initializer = (img_initializer, labels_initializer)
        else:
            input_shapes = (shapes[1:],)
            initializer = (img_initializer,)

        self.buffer = Buffer(*input_shapes, size=self.buffer_size, dtype=np.float32, initializer=initializer)

        return tf.data.Dataset.from_generator(lambda: self.buffer.shuffle().repeat(), buffer_dtypes,
                                              output_shapes=input_shapes)

    # TODO: Remove folder name
    def train_model(self, x, batch_size, alpha_scheduler=None, learning_rate_scheduler=None, folder_name=None,
                    SAVE_EPOCH=1, seed_noise=None, seed_labels=None):
        assert self.optimizer  # You must first compile the model

        if isinstance(x, (tuple, list)):
            X_train = x[0]
        else:
            X_train = x

        self.batch_size = batch_size

        alpha_scheduler = alpha_scheduler or (lambda _: 1.0)
        learning_rate_scheduler = learning_rate_scheduler or (lambda _: self.optimizer._lr)

        # Create the dataset object
        dataset = tf.data.Dataset.from_tensor_slices(x).shuffle(X_train.shape[0]).batch(batch_size, drop_remainder=True)
        if self.buffer_size:
            # We shuffled the buffer dataset erlier when created the buffer generator
            buffer_dataset = self._get_buffer_dataset(X_train.shape).batch(batch_size // 2, drop_remainder=True)
            dataset = tf.data.Dataset.zip((dataset, buffer_dataset))

        if self.tpu_strategy:
            train_iterator = self.tpu_strategy.make_dataset_iterator(dataset)
            train_iterator_init = train_iterator.initialize()
            train_samples = next(train_iterator)
        else:
            train_iterator = dataset.make_initializable_iterator()
            train_iterator_init = train_iterator.initializer
            train_samples = train_iterator.get_next()

        if self.tpu_strategy:
            buffer_values_replica, dist_train_gen_replica, dist_train_dis_replica = self.tpu_strategy.experimental_run_v2(
                self.train_step, args=(train_samples,))
            dist_train_gen, dist_train_dis = dist_train_gen_replica.values, dist_train_dis_replica.values
        else:
            buffer_values, dist_train_gen, dist_train_dis = self.train_step(train_samples)

        # Initialize unitialized variables only
        all_variables = tf.global_variables()
        uninit_variables = [var for var in all_variables if not self.sess.run(tf.is_variable_initialized(var))]
        self.sess.run(tf.variables_initializer(uninit_variables))

        # Used to track training progress
        inputs = [seed_noise, seed_labels] if self.labels_emb_size else seed_noise
        generated_images = self.generator(inputs, training=False)[self.stage_num]

        losses = []
        for epoch in range(self.train_stages[self.stage_num]['train_epochs']):
            print('\n  Processing epoch: {} =========================================='.format(epoch))

            start = time.time()

            # Set up the transition coefficient with `alpha_scheduler`
            new_alpha = alpha_scheduler(epoch)  # *0 + 1
            self.sess.run(self.alpha.assign(new_alpha))

            # A tricky way to set up the learning rate on the fly during training with `learning_rate_scheduler`
            new_lr = learning_rate_scheduler(epoch)
            self.sess.run(self.learning_rate.assign(new_lr))

            # Train loop
            self.sess.run(train_iterator_init)
            train_steps = X_train.shape[0] // (
                    batch_size * (self.dis_train_iters + self.gen_train_iters))

            # Set `proba` to 1 so that buffer stores every generated samples until it is full
            proba = self.buffer_store_proba if self.buffer_size and self.buffer.is_full else 1
            for step in range(train_steps):
                try:
                    # Disriminator training loop
                    for _ in range(self.dis_train_iters):
                        loss_d = self.sess.run(dist_train_dis)

                    # Generator training loop
                    for _ in range(self.gen_train_iters):
                        loss_g = self.sess.run(dist_train_gen)

                    if self.buffer_size and random.random() < proba:
                        self.buffer.store(*self.sess.run(buffer_values))

                except (StopIteration, tf.errors.OutOfRangeError):
                    break

            loss_d, loss_g = np.mean(loss_d), np.mean(loss_g)
            losses.append([loss_d, loss_g])
            print('    Epoch: {}; Alpha: {};, D_loss: {:.4}; G_loss: {:.4}'
                  .format(epoch, new_alpha, loss_d, loss_g))
            print("    Train Epoch time:  %.3f s" % (time.time() - start))

            if (epoch + 1) % SAVE_EPOCH == 0:
                # Save the weights
                self.save_weights('{}/weights'.format(folder_name))

                samples = self.sess.run(generated_images)
                fig = plot(samples, 10, 10, title='stage:{} epoch:{}'.format(self.stage_num, str(epoch).zfill(3)))
                plt.savefig('{}/progress/{}_{}_{}.png'
                            .format(folder_name, self.stage_num, self.gan_mode, str((epoch + 1)).zfill(3)),
                            bbox_inches='tight')
                plt.close(fig)

            fig = plt.figure()
            plt.plot(losses)
            plt.savefig('{}/{}_{}.jpeg'.format(folder_name, self.gan_mode, 'losses'))
            plt.close(fig)

    def save_weights(self, path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

        # Since the model is subclassed, we can only save weights with the specified `save_format` argument
        # But in case of TPUs using this kind of funcionalyty has not yet been implemented, so we have to
        # save our weights differently.

        # Get all the weights as the list of numpy arrays
        arrays = [var.values[0].eval(self.sess) for var in self.generator.trainable_variables]
        # And save them as one file
        np.savez('{}/generator'.format(path), arrays)

        # The same goes for the discriminator
        arrays = [var.values[0].eval(self.sess) for var in self.discriminator.trainable_variables]
        np.savez('{}/discriminator'.format(path), arrays)

    def load_weights(self, path):
        # We have to load numpy arrays with weights and reassining them to the corresponding tensors
        arrays = np.load('{}/generator.npz'.format(path), allow_pickle=True)
        for weight, tensor in zip(arrays['arr_0'], self.generator.trainable_weights):
            try:
                self.sess.run(tensor.assign(weight))
            except ValueError:
                pass

        arrays = np.load('{}/discriminator.npz'.format(path), allow_pickle=True)
        for weight, tensor in zip(arrays['arr_0'], self.discriminator.trainable_weights):
            try:
                self.sess.run(tensor.assign(weight))
            except ValueError:
                pass
