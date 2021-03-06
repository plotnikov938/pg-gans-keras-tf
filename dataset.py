import glob
import random

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import clip_value, resize_images_tf


def parse_rec(root):
    for child in root:
        yield child
        for sub_child in parse_rec(child):
            yield sub_child


# TODO: test as numpy
def make_square_tf(border):
    size = border[:, 2:] - border[:, :2]
    gap = (size[:, 0] - size[:, 1]) // 2

    border[gap > 0, 1::2] -= np.array([gap, -gap]).T[gap > 0]
    border[gap < 0, ::2] += np.array([gap, -gap]).T[gap < 0]
    #
    # border[gap < 0, 1::2] -= np.array([gap, -gap]).T[gap < 0]
    # border[gap > 0, ::2] += np.array([gap, -gap]).T[gap > 0]

    return border


def encode_breed(breeds):
    idxs = np.lexsort(breeds[None, :], axis=0)
    b = breeds[idxs]
    args = np.argwhere(b[1:] != b[:-1]).ravel() + 1

    for idx, split in enumerate(np.split(b, args)):
        split[:] = idx

    breeds[idxs] = b

    return breeds.astype(np.int32)


def parse_annotations(path_to_annts='annotations/'):
    """Parses the images annotation files in xml format.

      Arguments:
          path_to_annts: path where annotations is stored

      Returns:
          Pandas Dataframe object: `(num_imgs, num_data_fields)`.
      """

    # Open annotations
    annotations = []
    for filename in glob.iglob(path_to_annts + '**/*.xml', recursive=True):
        # Parse
        tree = ET.parse(filename)
        root = tree.getroot()

        pic_info = {}
        for child in parse_rec(root):
            pic_info[child.tag] = child.text

        pic_info['breed'] = pic_info['filename'].rsplit('_', 1)[0].lower()
        annotations.append(pic_info)

    # Create df from the list of dicts
    df = pd.DataFrame(annotations)

    # Drop useless columns
    df = df.drop(columns=fields_to_drop)

    # Cast coords of head and image size from str to int
    df[fields_borders + ['height', 'width']] = df[fields_borders + ['height', 'width']].astype('int')

    # Create breed_id separately for cats and dogs for a classification problem
    df['breed_id'] = None

    slice_dogs = df['name'] == 'dog'
    dog_breeds = df.loc[slice_dogs, 'breed'].values
    df.loc[slice_dogs, 'breed_id'] = encode_breed(dog_breeds)

    slice_cats = df['name'] == 'cat'
    cat_breeds = df.loc[slice_cats, 'breed'].values
    df.loc[slice_cats, 'breed_id'] = encode_breed(cat_breeds)

    return df


def save_annotations(path, df):
    np.savetxt(
        path + '.csv', df.values.astype('str'), fmt=','.join(['%s'] * 10),
        header=','.join(df.columns)
    )


def load_annotations(path):
    try:
        return pd.read_csv(path + '.csv')
    except FileNotFoundError:
        return None


def augmentation(img, size, crop_frac=0.7, flip_v=True, flip_h=True, rot90=True,
                 brightness_delta=.1, contrast_min_max=(0.5, 1.5), stddev_max=0.05):
        """

        :param img: (float)
            Data range from 0.0 to 1.0. [batch_size, height, width, channels]
        :param size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
          new size for the images.
        :param crop_frac: (float)
            The lower bound of cropping scale to randomly decreasing size of the source image
        :param brightness_delta:
        :param contrast_min_max:
        :param stddev_max:
        :return:
            Resized images. Data range from 0.0 to 1.0. [batch_size, size[0], size[1], channels]
        """

        # Random cropping with shifting
        if crop_frac:
            frac = tf.random.uniform([], minval=crop_frac)
            img_shapes = tf.shape(img)

            try:
                img_shapes_squared = tf.cast(tf.minimum(img_shapes, tf.reduce_min(img_shapes[:2])), tf.float32)
                crop_size = tf.cast(img_shapes_squared * frac, dtype=tf.int32)
                img = tf.image.random_crop(img, (crop_size[0], crop_size[1], img_shapes[-1]))
            except ValueError:
                # Supporting batching
                img_shapes_squared = tf.cast(tf.minimum(img_shapes[1:], tf.reduce_min(img_shapes[1:3])), tf.float32)
                crop_size = tf.cast(img_shapes_squared * frac, dtype=tf.int32)
                img = tf.image.random_crop(img, (img_shapes[0], crop_size[0], crop_size[1], img_shapes[-1]))

        # Random flipping
        if flip_h:
            img = tf.image.random_flip_left_right(img)
        if flip_v:
            # TODO: Прочекать, может быть не совсем полезная фича с точки зрения репрезентативности данных
            img = tf.image.random_flip_up_down(img)

        # Random rotation
        if rot90:
            img = tf.image.rot90(img, tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32))

        # TODO: for discrim train set range from -1 to 1
        if contrast_min_max:
            img = tf.image.random_contrast(img, *contrast_min_max)

        # Gaussian Noise
        if stddev_max:
            stddev = tf.random.uniform(shape=(), maxval=stddev_max)
            noise = tf.random.normal(shape=tf.shape(img), stddev=stddev)
            img += noise

        # Random_brightness
        if brightness_delta:
            delta = tf.random.uniform(shape=(), minval=-brightness_delta, maxval=brightness_delta)
            img = tf.clip_by_value(img - delta, 0.0, 1.0)

        return tf.image.resize(img, size)


def load_data_new(folder, size, frac=1.0, split=0.3, equal_sets=True, resize=False, mode=None, scale=.3):
    """
    :param size: A 1-D int32 tuple or list or np.array of 3 elements: `height, width, channels`.  The
      size for the images to produce.
    :param frac: (float)
        The fraction of samples from the total data used for training and testing
    :param split: (float)
        The fraction of samples used for testing. As the fraction of the total data, this can be computed
        with formula: 'frac' * 'split'
    :param equal_sets: (bool). Default true.
        Chose whether the length of the data for one label must be equal to the length of the data for another
        (Use in case of different data lengths)
    :param mode: (int)
        'cat': Loads only images of cats
        'dog': Loads only images of dogs
        None: Loads all images
    :param scale: (float)
        The value is used to scale up boundary box of an animal's head on an image
    """
    # Get num of the channels
    channels = size[-1]
    size = size[:2]
    # Clip the parameters value
    frac, split = np.clip([frac, split], 0.0, 1.0)

    df = load_annotations(folder + 'annotations')
    if df is None:
        df = parse_annotations(folder + ants_folder)
        save_annotations(folder + 'annotations', df)

    # Mask the dataset
    if mode is not None:
        df = df[df["name"] == mode]

    # Turn specie name into '0-1' label
    df["name"] = (df["name"] == 'cat')

    array = df[['filename'] + fields_borders].values

    # Pre-process borders
    borders = make_square_tf(array[:, 1:])
    borders[:, 2:] -= borders[:, :2]

    # Increase the borders size by scale factor for further augmentation with less data loss
    delta = (borders[:, -1] * scale).astype('int')[:, None]
    borders[:, :2] -= delta
    borders[:, :2] = np.maximum(borders[:, :2], 0) + 1
    borders[:, 2:] += delta * 2
    borders[:, 2:] = np.minimum(borders[:, 2:], df[['height', 'width']] - borders[:, :2]) - 1

    borders = tf.constant(borders.astype('int'))

    # import glob
    # print(folder + imgs_folder + '/*.{}'.format('jpeg'))
    # imagepaths = glob.glob(folder + imgs_folder + array[:, 0] + '/*.{}'.format('jpeg'), recursive=True)
    # input(imagepaths)

    # Convert imagepaths to Tensor
    imagepaths = (folder + imgs_folder + array[:, 0]).astype('str')
    imagepaths = tf.constant(imagepaths)

    def _func(imagepath, borders):
        image_raw = tf.read_file(imagepath)
        image = tf.image.decode_and_crop_jpeg(image_raw, borders, channels)
        if resize:
            image = tf.image.resize(image, size)
        image_casted = tf.cast(image / 255, tf.float32)

        return image_casted

    dataset = tf.data.Dataset.from_tensor_slices((imagepaths, borders))
    dataset = dataset.map(_func).repeat(1)

    iterator = dataset.make_one_shot_iterator()
    next_example = iterator.get_next()

    def generator():
        with tf.Session() as sess:
            while True:
                try:
                    yield sess.run(next_example)

                except tf.errors.OutOfRangeError:
                    break

    # Split dataset into train and test samples
    if mode is None:
        mode = df['name'] == 0
        dogs_rnd_ids = df[mode].index.values
        cats_rnd_ids = df[~mode].index.values
        np.random.shuffle(dogs_rnd_ids)
        np.random.shuffle(cats_rnd_ids)
        max_size = int(min(len(dogs_rnd_ids), len(cats_rnd_ids))*frac)
        split = int(max_size*split)

        if equal_sets:
            dogs_rnd_ids = dogs_rnd_ids[:max_size]
            cats_rnd_ids = cats_rnd_ids[:max_size]

        train_rnd_ids = np.concatenate([dogs_rnd_ids[split:], cats_rnd_ids[split:]])
        test_rnd_ids = np.concatenate([dogs_rnd_ids[:split], cats_rnd_ids[:split]])
    else:
        split = int(len(df)*split)
        train_rnd_ids, test_rnd_ids = np.split(range(len(df)), [-split])

    if resize:
        cashe_imgs = np.array([*generator()])

        return ((cashe_imgs[train_rnd_ids],
                 df[['name', 'breed_id']].iloc[train_rnd_ids].values),
                (cashe_imgs[test_rnd_ids],
                 df[['name', 'breed_id']].iloc[test_rnd_ids].values))
    else:
        cashe_imgs = [*generator()]

        return (([cashe_imgs[i] for i in train_rnd_ids],
                 df[['name', 'breed_id']].iloc[train_rnd_ids].values),
                ([cashe_imgs[i] for i in test_rnd_ids],
                 df[['name', 'breed_id']].iloc[test_rnd_ids].values))


def load_data(folder, size, frac=1.0, split=0.3, equal_sets=True, resize=False, mode=None, scale=.3):
    """
    :param size: A 1-D int32 tuple or list or np.array of 3 elements: `height, width, channels`.  The
      size for the images to produce.
    :param frac: (float)
        The fraction of samples from the total data used for training and testing
    :param split: (float)
        The fraction of samples used for testing. As the fraction of the total data, this can be computed
        with formula: 'frac' * 'split'
    :param equal_sets: (bool). Default true.
        Chose whether the length of the data for one label must be equal to the length of the data for another
        (Use in case of different data lengths)
    :param mode: (int)
        'cat': Loads only images of cats
        'dog': Loads only images of dogs
        None: Loads all images
    :param scale: (float)
        The value is used to scale up boundary box of an animal's head on an image
    """
    # Get num of the channels
    channels = size[-1]
    size = size[:2]
    # Clip the parameters value
    frac, split = np.clip([frac, split], 0.0, 1.0)

    df = load_annotations(folder + 'annotations')
    if df is None:
        df = parse_annotations(folder + ants_folder)
        save_annotations(folder + 'annotations', df)

    # Mask the dataset
    if mode is not None:
        df = df[df["name"] == mode]

    # Turn specie name into '0-1' label
    df["name"] = (df["name"] == 'cat')

    array = df[['filename'] + fields_borders].values

    # Pre-process borders
    borders = make_square_tf(array[:, 1:])
    borders[:, 2:] -= borders[:, :2]

    # Increase the borders size by scale factor for further augmentation with less data loss
    delta = (borders[:, -1] * scale).astype('int')[:, None]
    borders[:, :2] -= delta
    borders[:, :2] = np.maximum(borders[:, :2], 0) + 1
    borders[:, 2:] += delta * 2
    borders[:, 2:] = np.minimum(borders[:, 2:], df[['height', 'width']] - borders[:, :2]) - 1

    borders = tf.constant(borders.astype('int'))

    # import glob
    # print(folder + imgs_folder + '/*.{}'.format('jpeg'))
    # imagepaths = glob.glob(folder + imgs_folder + array[:, 0] + '/*.{}'.format('jpeg'), recursive=True)
    # input(imagepaths)

    # Convert imagepaths to Tensor
    imagepaths = (folder + imgs_folder + array[:, 0]).astype('str')
    imagepaths = tf.constant(imagepaths)

    def _func(imagepath, borders):
        image_raw = tf.read_file(imagepath)
        image = tf.image.decode_and_crop_jpeg(image_raw, borders, channels)
        if resize:
            image = tf.image.resize(image, size)
        image_casted = tf.cast(image / 255, tf.float32)

        return image_casted

    dataset = tf.data.Dataset.from_tensor_slices((imagepaths, borders))
    dataset = dataset.map(_func).repeat(1)

    iterator = dataset.make_one_shot_iterator()
    next_example = iterator.get_next()

    def generator():
        with tf.Session() as sess:
            while True:
                try:
                    yield sess.run(next_example)

                except tf.errors.OutOfRangeError:
                    break

    # Split dataset into train and test samples
    if mode is None:
        mode = df['name'] == 0
        dogs_rnd_ids = df[mode].index.values
        cats_rnd_ids = df[~mode].index.values
        np.random.shuffle(dogs_rnd_ids)
        np.random.shuffle(cats_rnd_ids)
        max_size = int(min(len(dogs_rnd_ids), len(cats_rnd_ids))*frac)
        split = int(max_size*split)

        if equal_sets:
            dogs_rnd_ids = dogs_rnd_ids[:max_size]
            cats_rnd_ids = cats_rnd_ids[:max_size]

        train_rnd_ids = np.concatenate([dogs_rnd_ids[split:], cats_rnd_ids[split:]])
        test_rnd_ids = np.concatenate([dogs_rnd_ids[:split], cats_rnd_ids[:split]])
    else:
        split = int(len(df)*split)
        train_rnd_ids, test_rnd_ids = np.split(range(len(df)), [-split])

    from collections import namedtuple
    Map = namedtuple('dataset', ['train', 'test'])

    if resize:
        cashe_imgs = np.array([*generator()])

        return Map(Dataset(cashe_imgs[train_rnd_ids],
                           df[['name', 'breed_id']].iloc[train_rnd_ids].values,
                           size), \
                   Dataset(cashe_imgs[test_rnd_ids],
                           df[['name', 'breed_id']].iloc[test_rnd_ids].values,
                           size))
    else:
        cashe_imgs = [*generator()]

        return Map(Dataset([cashe_imgs[i] for i in train_rnd_ids],
                           df[['name', 'breed_id']].iloc[train_rnd_ids].values,
                           size), \
                   Dataset([cashe_imgs[i] for i in test_rnd_ids],
                           df[['name', 'breed_id']].iloc[test_rnd_ids].values,
                           size))


def preprocess_dataset(train_data, test_data=None, size=None):
    """Preprocesses an input images dataset

    The function concatenates train and test data then standardizes the images in range [-1, 1],
    resizes them, if necessary, and applies one-hot tranformation to the conctenated labels.

    Args:
        train_data: A tuple or list containing train images labels.
            Images should be a 4-D `numpy.ndarray` of shape `[train_batch, height, width, channels]`.
            Labels should be a 1-D `numpy.ndarray` of shape `[train_batch]`.
        test_data (optional): A tuple or list containing test images and labels. Default to None.
            Images and labels have the same dtype and shapes as the train_date.
            If None than test data will not be used.
        size (optional): A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The initial
            size for the images in the train dataset. By default, the size of the images does not change.

    Returns:
        Prepared for training and evaluation dataset with train images
        of shapes `[train_batch + test_batch, height, width, channels]` in range [-1, 1] and one-hoted labels
    """

    X_train, y_train = train_data
    if test_data:
        X_test, y_test = test_data
        X_train, y_train = np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test))

    if len(X_train.shape) == 3:
        X_train = X_train[..., None]

    # Rescale -1 to 1
    X_train = (((X_train - X_train.min()) / X_train.max()) * 2 - 1)
    if size:
        X_train = resize_images_tf(X_train, size)

    y_train = tf.keras.utils.to_categorical(y_train, y_train.max() + 1)

    return X_train, y_train


class Dataset:
    def __init__(self, *data):
        self.storage = data
        self.data_aug = self.storage[0].copy()

        self.buf_arr = np.arange(len(self.storage[0]) * 2)

        self.batch_size = 1
        self.indexes = []
        self.repeat_times = -1
        self.samples_ph = None
        self.sample = None
        self._aug_flag = None
        self._sfl_flag = None
        self.batch_size = 1
        self._batch_shape = ()

    def shuffle(self):
        # Used to determine the order of method calling during batching
        if self._sfl_flag is None:
            self._sfl_flag = self.repeat_times

        if self._aug_flag is None:
            # Shuffle only the first half of the index buffer which one will
            # be using to return shuffled not augmented data
            np.random.shuffle(self.buf_arr[:len(self.buf_arr) // 2])
        else:
            np.random.shuffle(self.buf_arr)

        return self

    def batch(self, batch_size=None):
        if batch_size:
            batch_size = abs(int(batch_size))
            self.batch_size = batch_size
            self._batch_shape = (batch_size,)
        else:
            self.batch_size = 1
            self._batch_shape = ()

        return self

    def repeat(self, times=-1):
        self.repeat_times = times

        return self

    def augment(self):
        if self._aug_flag is None:
            self._aug_flag = self.repeat_times

        # Implement your augmentation func here

        return self

    def __iter__(self):
        counter = 0

        while True:
            if counter == self.repeat_times:
                break

            if counter != 0:
                # Augment data in the cache for each `repeat` iteration if self.augment()
                # was called after self.repeat()
                if self.repeat_times == self._aug_flag:
                    self.augment()

                # Shuffle data in the cache for each `repeat` iteration if self.shuffle()
                # was called after self.repeat()
                if self.repeat_times == self._sfl_flag:
                    self.shuffle()

            if self._aug_flag is None:
                upper_bound = len(self.buf_arr) // 2 // self.batch_size * self.batch_size
                self.indexes = self.buf_arr[:upper_bound].reshape(-1, *self._batch_shape)

                for idxs in self.indexes:
                    yield tuple(arr[idxs] for arr in self.storage)
            else:
                upper_bound = len(self.buf_arr) // self.batch_size * self.batch_size
                self.indexes = self.buf_arr[:upper_bound].reshape(-1, *self._batch_shape)

                for idxs in self.indexes:
                    yield tuple(np.concatenate([self.storage[0], self.data_aug], axis=0)[idxs],
                                *(np.tile(arr, 2)[idxs] for arr in self.storage[1:]))

            counter += 1

        # Reset all the properties
        self._aug_flag, self._sfl_flag, self.repeat_times = None, None, -1


class Buffer(Dataset):
    """A simple circular buffer used to store previously generated data

    Args:
        *shape: A tuples or lists of integer values with stored data shapes.
        size (int, optional): A size of the buffer. Default to None.
            If not specified, the outer dimention in the first `shape` tuple will be used as the `size`.
        dtype (optional): A tuple or list of stored data dtypes. Default to `numpy.zeros`.
            Specifies the dtype of stored data.
        initializer (optional): A tuple or list of functions with the signature `(shape, dtype=None)`.
            Initializers for the arrays with the stored data.
    """

    def __init__(self, *shape, size=None, dtype='float32', initializer=np.zeros):

        self.shape = shape
        self.size = size or shape[0][0]
        self.dtype = dtype
        if size:
            self.shape = tuple((self.size, *shape) for shape in self.shape)

        self.ptrs = np.arange(0, size, 1, dtype=np.int32)
        self._len = 0

        # Check if the `initializer` needs to be broadcasted to match the length of the `shape`
        if not hasattr(initializer, '__iter__'):
            initializer = (initializer,)
        if any(isinstance(i, (list, tuple)) for i in self.shape) and len(initializer) == 1:
            initializer = initializer * len(self.shape)

        # Check if the `dtype` needs to be broadcasted to match the length of the `shape`
        if not hasattr(dtype, '__iter__'):
            self.dtype = (self.dtype,)
        if any(isinstance(i, (list, tuple)) for i in self.shape) and len(self.dtype) == 1:
            self.dtype = self.dtype * len(self.shape)

        super(Buffer, self).__init__(
            *[init(shape=shape, dtype=dtype) for init, shape, dtype in zip(initializer, self.shape, self.dtype)])
        # super(Buffer, self).__init__(initializer(shape=shape, dtype=dtype))

    @classmethod
    def from_data(cls, *data):
        """An alternative constructor for the 'Buffer' object.

        Allows to create buffer from already existing data.

        Args:
            *data: A tuples or lists of integer values with stored data.

        Returns:
            A 'Buffer' object.
        """

        if any(isinstance(i, (list, tuple)) for i in data):
            data = data[0]

        buffer = cls(*(array.shape for array in data), dtype=tuple(array.dtype for array in data))
        for buf, array in zip(buffer.storage, data):
            buf[:] = array

        return buffer

    def store(self, *input_data):
        """A function that allows to store the `input_data` into the circular buffer.

        Args:
            input_data (numpy.ndarray): A numpy array of the inputs data.

        Raises:
            AssertionError: If the shape of the `input_data` is not equal to the specified or
                if the length of the `input_data` is less then or equal to the max buffer capacity.

        Returns:
            None
        """

        # Store the input data
        input_data_len = input_data[0].shape[0]
        ptrs = self.ptrs[:input_data_len]
        for shape, storage, data in zip(self.shape, self.storage, input_data):

            assert shape[1:] == data.shape[1:]  # The shape of the passed data must be equal to the specified
            assert shape[0] >= data.shape[0]  # The length of the input data must be less than or equal to the max buffer capacity

            storage[ptrs] = data

        # Roll the pointers array backward by the size of the stored data
        self.ptrs = np.roll(self.ptrs, -input_data_len)

        if not self.is_full:
            self._len += input_data_len

    # TODO: Deprecated (maybe)
    def store_with_prob(self, *input_data, prob_to_store=1.0):
        """A function that allows to store the `input_data` into the circular buffer with a certain probability.

        Args:
            input_data (numpy.ndarray): A numpy array of the inputs data.
            prob_to_store (float in range [0.0, 1.0], optional): A probability that determines
                whether or not to store the `input_data`. Default to 1.0.
                If equal to 0.0 than doesn't store anything, if equal to 1.0 than store everything.

        Returns:
            None
        """

        # Clip the probability value
        prob_to_store = clip_value(prob_to_store, 0.0, 1.0)

        if random.random() < prob_to_store:
            self.store(*input_data)

    def __len__(self):
        return min(self._len, self.shape[0][0])

    @property
    def is_full(self):
        """Get a boolean value indicating the buffer is full."""

        if self._len >= self.shape[0][0]:
            return True
        else:
            return False


if __name__ == "__main__":
    # TODO: Argpars
    raise NotImplementedError

    folder = 'dataset/'
    save_to = 'dataset/'
    new_size = (128, 128)

    imgs_folder = 'images/'
    ants_folder = 'annotations/'
    fields_borders = ['ymin', 'xmin', 'ymax', 'xmax']
    fields_to_drop = [
        'annotation', 'bndbox', 'database', 'source', 'folder', 'image', 'size', 'object',
        'difficult', 'segmented', 'truncated', 'occluded',
        'depth', 'pose'
    ]
