import os
import glob
import argparse
import warnings


import numpy as np
import tensorflow as tf

from utils import Scheduler, make_gif, empty_context_mgr, get_config
from dataset import resize_images_tf, preprocess_dataset
from models import GAN_PG


def train():
    tf.keras.backend.clear_session()

    with strategy.scope() if use_tpu else empty_context_mgr():

        if use_tpu:
            sess_config = tf.ConfigProto()
            sess_config.allow_soft_placement = True
            cluster_spec = resolver.cluster_spec()
            if cluster_spec:
                sess_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
        else:
            sess_config = None

        model = GAN_PG(train_stages, channels, latent_size=latent_size, use_wscale=use_wscale, labels_emb_size=labels_emb_size, gan_mode=gan_mode,
                       mixup_alpha=mixup,
                       drop_rate=drop_rate, batch_norm=batch_norm, pixel_norm=pixel_norm, dis_train_iters=dis_train_iters,
                       gen_train_iters=gen_train_iters, buffer_size=buffer_size, buffer_epoch_depth=buffer_epoch_depth)

        # Train
        for stage in train_stages:
            # Get training stage num
            stage_num = train_stages.index(stage)

            print('\nProcessing stage: {}  with the size {} =========================================='.format(
                stage_num, stage['size']))

            model(stage_num, training=True, tpu_strategy=strategy, resolver=resolver, config=sess_config)

            # TODO: Restore the model
            if restore:
                model.load_weights('{}/weights'.format(folder))
                print('The model has been sucsessfully restored!')

            # Compile the model
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
            model.compile_model(optimizer=optimizer, loss=gan_mode)

            # Resize train samples to the specifed stage size
            Images_train = resize_images_tf(X_train, stage["size"][:-1], sess=model.sess)

            # Define schedulers
            alpha_scheduler = Scheduler(stage['train_epochs'], [0, 0.5], [0, 1.0])
            learning_rate_scheduler = None

            try:
                os.makedirs(folder)
            except FileExistsError:
                pass

            # Train the model for the train_stage
            inputs = (Images_train, y_train) if conditional else Images_train
            model.train_model(inputs, batch_size,
                              alpha_scheduler=alpha_scheduler,
                              learning_rate_scheduler=learning_rate_scheduler,
                              folder_name='{}/{}'.format(folder, stage_num), SAVE_EPOCH=5, seed_noise=seed_noise,
                              seed_labels=seed_labels
                              )

    make_gif(glob.iglob('{}/progress/*.png'.format(folder)),
             '{}/{}_{}.gif'.format(folder, gan_mode, 'progress'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', type=str, default='./config.py', help='a path to the config.py')

    args = parser.parse_args()

    # load config
    config = get_config(args.path_config)

    # Load the dataset
    if config['dataset'] is 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif config['dataset'] is 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Load your custom dataset here
    else:
        raise NotImplementedError("Load your custom dataset here")

    X_train, y_train = preprocess_dataset((X_train, y_train), (X_test, y_test))
    channels = X_train.shape[-1]

    labels_emb_size = y_train.shape[-1] if config['conditional'] else None

    # Set seed parameters to control the training process
    seed_noise = np.random.normal(size=(y_train.shape[-1] * 10, config['latent_size'])).astype(np.float32)
    seed_labels = np.tile(np.arange(y_train.shape[-1])[:, None], (1, 10)).reshape(-1)
    seed_labels = tf.keras.utils.to_categorical(seed_labels, y_train.shape[-1]).astype(np.float32)

    use_buffer = bool(config['buffer_size'])

    if config['use_tpu']:
        if config['buffer_size']:
            warnings.warn('The using of the buffer on TPUs is not implemented')
            config['buffer_size'] = 0

        assert os.environ[
            'COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'
        assert float(
            '.'.join(tf.__version__.split('.')[:2])) >= 1.14, 'Make sure that Tensorflow version is at least 1.14'

        TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER)
        tf.contrib.distribute.initialize_tpu_system(resolver)
        strategy = tf.contrib.distribute.TPUStrategy(resolver)
    else:
        resolver, strategy = None, None

    # Run training
    train()
