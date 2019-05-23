import tensorflow as tf
import numpy as np

from tensorflow.python.keras.layers import Layer


class InterpBilinear(Layer):
    def __init__(self, size, *args, old_size=None, **kwargs):
        super(InterpBilinear, self).__init__()

        self.size = size
        self.channels = size[-1]
        self.old_size = old_size

    def call(self, inputs, *args, **kwargs):
        if len(inputs.shape) == 2:
            if self.old_size is None:
                side = tf.cast(np.sqrt(int(inputs.shape[-1]) // self.channels), tf.int32)
                self.old_size = (side, side, self.channels)

            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], *self.old_size))

        return tf.image.resize_bilinear(inputs, self.size[:2], align_corners=True)


class InterpBicubic(Layer):
    def __init__(self, size, *args, old_size=None, **kwargs):
        super(InterpBicubic, self).__init__()

        self.size = size
        self.channels = size[-1]
        self.old_size = old_size

    def call(self, inputs, *args, **kwargs):
        if len(inputs.shape) == 2:
            if self.old_size is None:
                side = tf.cast(np.sqrt(int(inputs.shape[-1]) // self.channels), tf.int32)
                self.old_size = (side, side, self.channels)

            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], *self.old_size))

        return tf.image.resize_bicubic(inputs, self.size[:2], align_corners=True)


class InterpNearestUpscale(Layer):
    def __init__(self, *args, scale=2, channels=1, **kwargs):
        super(InterpNearestUpscale, self).__init__()

        assert isinstance(scale, int) and scale >= 1

        self.scale = scale
        self.channels = channels

    def call(self, inputs, *args, **kwargs):
        if self.scale == 1:
            return inputs

        if len(inputs.shape) == 2:
            side = tf.cast(np.sqrt(int(inputs.shape[-1]) // self.channels), tf.int32)
            self.old_size = (side, side, self.channels)

            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], *self.old_size))

        shape = inputs.shape
        inputs = tf.reshape(inputs, [-1, shape[1], 1, shape[2], 1, shape[3]])
        inputs = tf.tile(inputs, [1, 1, self.scale, 1, self.scale, 1])
        outputs = tf.reshape(inputs, [-1, shape[1] * self.scale, shape[2] * self.scale, shape[3]])

        return outputs


class InterpNearestDownscale(Layer):
    def __init__(self, *args, scale=2, channels=1, **kwargs):
        super(InterpNearestDownscale, self).__init__()

        assert isinstance(scale, int) and scale >= 1

        self.scale = scale
        self.channels = channels

    def call(self, inputs, *args, **kwargs):
        if self.scale == 1:
            return inputs

        if len(inputs.shape) == 2:
            side = tf.cast(np.sqrt(int(inputs.shape[-1]) // self.channels), tf.int32)
            self.old_size = (side, side, self.channels)

            inputs = tf.reshape(inputs, (tf.shape(inputs)[0], *self.old_size))

        ksize = [1, self.scale, self.scale, 1]
        return tf.nn.avg_pool(inputs, ksize=ksize, strides=ksize, padding='VALID', data_format='NHWC')
