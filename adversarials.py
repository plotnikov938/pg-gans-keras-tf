import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.ops import math_ops


class BCE(Layer):
    """Computes the binary cross entropy loss from logits with labels broadcasting.

    Args:
      label_smoothing: If greater than `0` then one-sided smoothing will be applied to the labels.
      eps: Used for safe calculations.
    """

    def __init__(self,
                 label_smoothing=0,
                 eps=1e-10,
                 **kwargs):
        super(BCE, self).__init__(**kwargs)

        self.label_smoothing = label_smoothing
        self.eps = eps

    def call(self, logits, label):
        """Computes the binary cross entropy loss from logits with label broadcasting.

        Args:
            logits: A `Tensor` of type `float32` or `float64`.
            label: A `Tensor` of the same type as `logits`.

        Returns:
            A `Tensor` of the same type as `logits` with the binary cross entropy loss.
        """

        if self.label_smoothing > 0:
            label = (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        relu_logits = -tf.minimum(-logits, 0.0)
        neg_abs_logits = tf.minimum(-logits, logits)
        result = math_ops.add(relu_logits,
                              math_ops.log1p(math_ops.exp(neg_abs_logits)))

        if label == 1:
            result = math_ops.add(result, -logits)
        elif label != 0.0:
            result = math_ops.add(result, -logits * label)

        return tf.reduce_mean(result)


class AdversarialLosses(Layer):
    """Computes the specified adversarial loss.

    Args:
        mode (obj: str): A string specifying  the type of loss to be calculated.
            Allowed values: `gan`, `wgan`, `hinge`, `lsgan`, `d2gan`.

            All of the above losses except for `d2gan` could be used both with gradient penalty
            by specifing postfix `-lp` or  `-pg` and with relativistic gan
            approach by specifing prefix `r-` or  `ra-` (e.g. `ra-gan-lp`)

        label_smoothing: If greater than `0` then one-sided smoothing will be applied to the labels.

        Resources:
            - 'gan': [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
            - `wgan`: [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
            - `hinge`: [Hierarchical Implicit Models and Likelihood-Free Variational Inference]
                (https://arxiv.org/abs/1702.08896)
            - `lsgan`: [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)
            - `d2gan`: [Dual Discriminator Generative Adversarial Nets](https://arxiv.org/abs/1709.03831)
            - 'r-', 'ra-': [The relativistic discriminator: a key element missing from standard GAN]
                (https://arxiv.org/abs/1807.00734)
            - '-lp': [On the regularization of Wasserstein GANs](https://arxiv.org/abs/1709.08894)
            - '-gp': [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
    """

    def __init__(self, *args, mode='gan', label_smoothing=0, **kwargs):
        super(AdversarialLosses, self).__init__(*args, **kwargs)

        self.mode = mode
        self.label_smoothing = label_smoothing

        self.gp, self.lp = False, False
        if self.mode.endswith('-lp'):
            self.mode = self.mode[:-3]
            self.lp = True
        if self.mode.endswith('-gp'):
            self.mode = self.mode[:-3]
            self.gp = True

    def call(self, logits_real, logits_fake, *args,
             logits_real_2=None, logits_fake_2=None, lam=10, alpha=1.0, discriminator=None,
             beta=1.0, samples_real=None, samples_fake=None, **kwargs):
        """Computes the specified adversarial loss.

        Args:
            logits_real: A 2-D Tensor  of shape `[batch, lodits_dim]` with logits computed for real data.
            logits_fake: A 2-D Tensor  of shape `[batch, lodits_dim]` with logits computed for generated data.
            logits_real_2 (optional): A 2-D Tensor  of shape `[batch, lodits_dim]` with logits computed
                for real data by second discriminator. Used for `d2gan`. Defaults to None.
            logits_fake_2 (optional): A 2-D Tensor  of shape `[batch, lodits_dim]` with logits computed
                for generated data by second discriminator. Used for `d2gan`. Defaults to None.
            lam (value): A value used to compute loss with `-gp` and `-lp'. Defaults to 10.
            alpha (float): A value used to compute loss with `-gp` and `-lp'. Defaults to 1.0.
            discriminator (obj: function, optional):  The same function that computes `logits_real` and `logits_fake`.
                Used for `-gp` and `-lp' GAN. Defaults to None.
            beta (float): A value used to compute loss for `d2gan`. Defaults to 1.0.
            samples_real: Real samples used to compute logits for the loss with `-gp` and `-lp'
            samples_fake: Fake samples used to compute logits for the loss with `-gp` and `-lp'

        Returns:
            Tuple (loss_generator, loss_discrim) both with the shape [batch, 1]
        """

        if self.mode.startswith('r-'):
            logits_real, logits_fake = (logits_real - logits_fake,
                                        logits_fake - logits_real)
        elif self.mode.startswith('ra-'):
            logits_real, logits_fake = (logits_real - tf.reduce_mean(logits_fake),
                                        logits_fake - tf.reduce_mean(logits_real))

        if self.mode == 'gan':
            self.loss_generator = BCE(self.label_smoothing)(logits_fake, 1)
            self.loss_discrim = BCE(self.label_smoothing)(logits_fake, 0) + \
                                BCE(self.label_smoothing)(logits_real, 1)

        elif 'wgan' in self.mode:
            self.loss_generator = -tf.reduce_mean(logits_fake)
            self.loss_discrim = tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)

        elif self.mode == 'began':
            print("`BEGAN` hasn't yet been properly implemented")
            raise NotImplementedError

            logits_real, logits_fake = tf.nn.sigmoid(logits_real), tf.nn.sigmoid(logits_fake)
            # a = -tf.nn.tanh((tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)))
            # self.loss_generator = -tf.nn.tanh(tf.reduce_mean(logits_fake))
            # self.loss_discrim = tf.nn.tanh(tf.reduce_mean(logits_fake))*a - \
            #                     tf.nn.tanh(tf.reduce_mean(logits_real))
            self.loss_generator = -tf.reduce_mean(tf.math.log(logits_fake + 1e-8))
            k = -tf.reduce_mean(tf.math.log(logits_real + 1e-8) + tf.math.log(1 - logits_fake + 1e-8)) / self.loss_generator
            k = tf.clip_by_value(tf.stop_gradient(k), 1e-6, 1)
            self.loss_discrim = -tf.reduce_mean(tf.math.log(logits_real + 1e-8)) - tf.reduce_mean(tf.math.log(1 - logits_fake + 1e-8))*k

        elif self.mode == 'lsgan':
            self.loss_generator = tf.reduce_mean((logits_fake - 1.0) ** 2)# / 2
            self.loss_discrim = (tf.reduce_mean((logits_real - 1.0) ** 2) +
                                 tf.reduce_mean(logits_fake ** 2)) / 2.

        elif self.mode == 'hinge':
            self.loss_generator = tf.reduce_mean(tf.nn.relu(1.0 - logits_fake))
            self.loss_discrim = tf.reduce_mean(tf.nn.relu(1.0 - logits_real)) + \
                                tf.reduce_mean(tf.nn.relu(1.0 + logits_fake))

        elif self.mode == 'd2gan':
            assert logits_real_2 is not None  # Pass `logits_real_2` as a parameter
            assert logits_fake_2 is not None  # Pass `logits_fake_2` as a parameter

            logits_real, logits_fake = tf.nn.softplus(logits_real), tf.nn.softplus(logits_fake)
            logits_real_2, logits_fake_2 = tf.nn.softplus(logits_real_2), tf.nn.softplus(logits_fake_2)

            self.loss_generator = tf.reduce_mean(-logits_fake + beta * tf.math.log(logits_fake_2))
            self.loss_discrim = tf.reduce_mean(-alpha * tf.math.log(logits_real) + logits_fake) + \
                                tf.reduce_mean(logits_real_2 - beta * tf.math.log(logits_fake_2))

        elif 'r-gan' in self.mode:
            self.loss_generator = BCE(self.label_smoothing)(logits_fake, 1)
            self.loss_discrim = BCE(self.label_smoothing)(logits_real, 1)

        elif 'ra-gan' in self.mode:
            self.loss_generator = (BCE(self.label_smoothing)(logits_fake, 1) +
                                   BCE(self.label_smoothing)(logits_real, 0))/2
            self.loss_discrim = (BCE(self.label_smoothing)(logits_real, 1) +
                                 BCE(self.label_smoothing)(logits_fake, 0))/2

        elif any(mode in self.mode for mode in ('r-lsgan', 'ra-lsgan')):
            self.loss_generator = tf.reduce_mean((logits_fake - 1) ** 2) + \
                                  tf.reduce_mean((logits_real + 1) ** 2)
            self.loss_discrim = tf.reduce_mean((logits_real - 1) ** 2) + \
                                tf.reduce_mean((logits_fake + 1) ** 2)

        elif any(mode in self.mode for mode in ('r-hinge', 'ra-hinge')):
            self.loss_generator = tf.reduce_mean(tf.nn.relu(1.0 - logits_fake)) + \
                                  tf.reduce_mean(tf.nn.relu(1.0 + logits_real))
            self.loss_discrim = tf.reduce_mean(tf.nn.relu(1.0 - logits_real)) + \
                                tf.reduce_mean(tf.nn.relu(1.0 + logits_fake))

        else:
            raise Exception()

        if self.lp or self.gp:
            assert discriminator is not None  # Pass `discriminator` as a parameter
            assert samples_real is not None  # Pass `samples_real` as a parameter
            assert samples_fake is not None  # Pass `samples_fake` as a parameter

            if not isinstance(samples_real, (list, tuple)):
                samples_real, samples_fake = [samples_real], [samples_fake]
                _discriminator = lambda x: discriminator(*x)
            else:
                _discriminator = discriminator

            interpolates = []
            batch_size = tf.shape(logits_real)[0]
            for _real, _fake in zip(samples_real, samples_fake):
                shape = [1] * (len(_real.shape) - 1)
                alpha = tf.random.uniform(shape=[batch_size, *shape], minval=0., maxval=1.)
                interpolates.append(alpha * (_real - _fake) + _fake)

            gradients = tf.gradients(_discriminator(interpolates), interpolates)

            gradient_penalties = []
            for gradient in gradients:

                if gradient is None:
                    continue

                grad_norm = tf.sqrt(tf.reduce_sum(tf.square(gradient + 1e-8), axis=1))
                if self.lp:
                    gradient_penalty = tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))
                else:
                    gradient_penalty = tf.reduce_mean(tf.square(grad_norm - 1.))

                gradient_penalties.append(lam * gradient_penalty)

            self.loss_discrim += sum(gradient_penalties)

        return self.loss_discrim, self.loss_generator