import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Wrapper
from tensorflow.python.keras import backend as K


def clip_value(a, a_min, a_max):
    """Clip (limit) the value.

        Given an interval, values outside the interval are clipped to
        the interval edges.  For example, if an interval of ``[0, 1]``
        is specified, values smaller than 0 become 0, and values larger
        than 1 become 1.

    Args:
        a (scalar): A value containing elements to clip.
        a_min (scalar): Minimum value.
        a_max (scalar): Maximum value.

    Returns:
        clipped_value (scalar): A value of `a` but where
            values < `a_min` are replaced with `a_min`, and those > `a_max`
            with `a_max`.
    """

    return a_min if a < a_min else a_max if a > a_max else a


def mixup_func(major_samples, minor_samples, alpha=0):
    """Mixup implementation is from here: https://www.inference.vc/mixup-data-dependent-data-augmentation/

    Args:
        major_samples: A Tensor of shape [batch_size, ...] or list of tensors with most preferred samples.
        minor_samples: A Tensor of same shape and dtype as `major_samples` or list of tensors with less preferred samples with a shape equal to the shape of the `major_samples` tensor.
        alpha(float): A value in range [0, 1] using to define beta distribution. If `alpha` is 0, than mixup is not used.

    Raises:
        AssertionError: If the shape of the tensor or the length of the list of tensors `major_samples`
            is not equal to the shape of the tensor or the length of the list of tensors `minor_samples`.

    Returns:
        If `alpha` is 0:
            A Tensor or list of Tensors `major_samples`
        else:
            A Tensor or list of Tensors  with mixed samples consisting of a combination of major_samples and minor_samples
            in the proportion based on the beta distribution
    """

    if isinstance(major_samples, (list, tuple)):
        assert len(major_samples) == len(minor_samples)  # The length of the lists must be the same
    else:
        major_samples, minor_samples = [major_samples], [minor_samples]

    assert major_samples[0].shape[1:] == minor_samples[0].shape[1:]  # The shapes of the tensors must be the same

    alpha = clip_value(alpha, 0.0, 1.0)

    if alpha == 0.0:
        return major_samples

    mixed_samples = []
    kumaraswamy = tfd.Kumaraswamy(alpha + 1.0, alpha)
    for major, minor in zip(major_samples, minor_samples):
        try:
            sample_shape = (tf.shape(major)[0], *[1]*(len(major.shape) - 1))
            mix_fraction = kumaraswamy.sample(sample_shape)
            mixed = mix_fraction * (major - minor) + minor
        except ValueError:
            mixed = major

        mixed_samples.append(mixed)

    return mixed_samples


# TODO: DONE: Arg sess, chunks, docs, new logic
def resize_images_tf(images, size, chunks=None, sess=None):
    """A function that resizes `images` to `size` using tensorflow bilinear interpolation.

    Args:
        images: A numpy array with shape (batch_size, height, width, channels).
        size: A list of 'int' or a numpy array of type 'int' containing the new size of images in format(height, width).
        chunks: An optional 'int'. Defaults to None.
            If the input array `images` is to large to fit in the memory for processing in one go, you can split
            the `images` into chunks, and than process each chunk separately.
        sess: An optional tf.Session() object. Defaults to None.
            If None, creates a new tf.Session() object inside the function.
            Otherwise uses specified tf.Session() object.

    Returns:
        A numpy array of type `float32` containing images with the new size.
    """

    import tensorflow as tf
    tf = tf.compat.v1

    need_close = False if sess else True
    sess = sess or tf.Session()
    chunks = chunks or 1

    # Split into chunks
    new_images = []
    ph = tf.placeholder(tf.float32, (None, None, None, None))
    img = tf.image.resize_bilinear(ph, size, align_corners=True)
    for image_chunk in np.array_split(images, chunks):
        new_images.append(sess.run(img, {ph: image_chunk}))

    if need_close:
        sess.close()

    return np.concatenate(new_images, axis=0)


def dropConnect(x, p):
    return tf.nn.dropout(x, rate=1 - p) * p


class Scheduler:
    """Scheduler of value.

    Takes as input current epoch number and return required value based on the linear interpolation
    for the given discrete datapoints `xp` and `fp`.

        Args:
            epochs (int): Total epochs number.
            xp: 1-D sequence of floats.
                The epochs-coordinates of the data points, must be increasing.
            fp: 1-D sequence of floats.
                The values-coordinates of the data points, same length as `xp`.
      """

    def __init__(self, epochs, xp=None, fp=None, *args, **kwargs):
        self.epochs = epochs
        self.xp = np.array(xp)*self.epochs
        self.fp = np.array(fp)

    def __call__(self, epoch, *args, **kwargs):
        """Returns value for the given epoch based on the linear interpolation

        Args:
            epoch (int): Current epoch value.

        Returns
            interpolated value (float or ndarray): The interpolated value for the given epoch.
        """

        return np.interp(epoch, self.xp, self.fp)


class PixelNormalization(tf.keras.layers.Layer):
    """Pixel normalization layer.

    Normalize the outputs of the previous layer across channel dimention.

    Args:
        axis (int): The axis that should be normalized
        epsilon (float): Small float added to avoid dividing by zero.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.
    """

    def __init__(self, epsilon=1e-8, axis=-1):
        super(PixelNormalization, self).__init__()
        self.epsilon = epsilon
        self.axis = axis

    def call(self, inputs, *args, **kwargs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=self.axis, keepdims=True) + self.epsilon)


# TODO: Refactor later
class DropConnectDense(Dense):
    def __init__(self, *args, **kwargs):
        self.prob = kwargs.pop('prob', 0.5)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True
        super(DropConnectDense, self).__init__(*args, **kwargs)

    def call(self, x, mask=None):
        if 0. < self.prob < 1.:
            self.kernel = K.in_train_phase(K.dropout(self.kernel, self.prob), self.kernel)
            self.b = K.in_train_phase(K.dropout(self.b, self.prob), self.b)

        # Same as original
        output = K.dot(x, self.W)
        if self.bias:
            output += self.b
        return self.activation(output)


# TODO: Refactor later
class DropConnect(Wrapper):
    def __init__(self, layer, prob=1., **kwargs):
        self.prob = prob
        self.layer = layer
        super(DropConnect, self).__init__(layer, **kwargs)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        if 0. < self.prob < 1.:
            self.layer.kernel = K.in_train_phase(K.dropout(self.layer.kernel, self.prob), self.layer.kernel)
            self.layer.bias = K.in_train_phase(K.dropout(self.layer.bias, self.prob), self.layer.bias)
        return self.layer.call(x)