import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import glob
# from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import tensorflow as tf

imgs_folder = 'images/'
ants_folder = 'annotations/'
fields_borders = ['ymin', 'xmin', 'ymax', 'xmax']
fields_to_drop = [
    'annotation', 'bndbox', 'database', 'source', 'folder', 'image', 'size', 'object',
    'difficult', 'segmented', 'truncated', 'occluded',
    'depth', 'pose'
]


def parse_rec(root):
    for child in root:
        yield child
        for sub_child in parse_rec(child):
            yield sub_child


def blur_cut_edges(img):
    # Paste image on white background with offset to center
    size = img.size
    background = Image.new('L', new_size, 255)
    offset = ((new_size[0] - size[0]) // 2, (new_size[1] - size[1]) // 2)
    background.paste(img, offset)

    # Blur params
    RADIUS = max(offset) // 2
    diam = RADIUS

    # Create blur mask
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)

    x1, y1 = size
    x0, y0 = 0, 0
    dec_1 = 1 if offset[0] else 0
    dec_2 = 1 if offset[1] else 0
    for d in range(diam + RADIUS):
        x1, y1 = x1 - dec_1, y1 - dec_2
        alpha = 255 if d < RADIUS else int(255 * (diam + RADIUS - d) / diam)
        draw.rectangle([x0, y0, x1, y1], outline=alpha)
        x0, y0 = x0 + dec_1, y0 + dec_2

    # Blur the image and paste blurred edges according to the mask
    blur = img.filter(ImageFilter.GaussianBlur(RADIUS))
    img.paste(blur, mask=mask)

    return img


def make_square(border):
    size = border[2:] - border[:2]
    gap = (size[0] - size[1]) // 2

    if gap > 0:
        border[1::2] -= [gap, -gap]
    elif gap < 0:
        border[::2] += [gap, -gap]


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


def image_preprocessing(filename, borders, size, convert="L"):
    img = Image.open(folder + imgs_folder + filename)

    # make_square(borders)
    cropped_img = img.crop(borders)

    converted_img = cropped_img.convert(convert)
    converted_img = thumbnail(converted_img, size, Image.ANTIALIAS)

    return converted_img

    # plt.imshow(pix, cmap='gist_gray', clim=(0, 255))
    # plt.colorbar()
    # plt.show()


def image_preprocessing_tf(filename, borders, size, convert="L"):
    img = Image.open(folder + imgs_folder + filename)

    make_square(borders)
    cropped_img = img.crop(borders)

    converted_img = cropped_img.convert(convert)
    converted_img = thumbnail(converted_img, size, Image.ANTIALIAS)

    # pix = np.array(converted_img)
    #
    # plt.imshow(pix, cmap='gist_gray', clim=(0, 255))
    # plt.colorbar()
    # plt.show()


def encode_breed(breeds):
    idxs = np.lexsort(breeds[None, :], axis=0)
    b = breeds[idxs]
    args = np.argwhere(b[1:] != b[:-1]).ravel() + 1

    for idx, split in enumerate(np.split(b, args)):
        split[:] = idx

    breeds[idxs] = b

    return breeds.astype(np.int32)


def parse_annotations(path_to_ants='annotations/'):
    """Parses the images annotation files in xml format.

      Arguments:
          path_to_ants: path where annotations is stored

      Returns:
          Pandas Dataframe object: `(num_imgs, num_data_fields)`.
      """

    # Open annotations
    annotations = []
    for filename in glob.iglob(path_to_ants + '**/*.xml', recursive=True):
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
            frac = tf.random_uniform([], minval=crop_frac)
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
            img = tf.image.rot90(img, tf.random_uniform(shape=[], minval=0, maxval=3, dtype=tf.int32))

        # TODO: for discrim train set range from -1 to 1
        if contrast_min_max:
            img = tf.image.random_contrast(img, *contrast_min_max)

        # Gaussian Noise
        if stddev_max:
            stddev = tf.random_uniform(shape=(), maxval=stddev_max)
            noise = tf.random_normal(shape=tf.shape(img), stddev=stddev)
            img += noise

        # Random_brightness
        if brightness_delta:
            delta = tf.random_uniform(shape=(), minval=-brightness_delta, maxval=brightness_delta)
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

        return Map(DatasetCatDog(cashe_imgs[train_rnd_ids],
                           df[['name', 'breed_id']].iloc[train_rnd_ids].values,
                           size), \
                   DatasetCatDog(cashe_imgs[test_rnd_ids],
                           df[['name', 'breed_id']].iloc[test_rnd_ids].values,
                           size))
    else:
        cashe_imgs = [*generator()]

        return Map(DatasetCatDog([cashe_imgs[i] for i in train_rnd_ids],
                           df[['name', 'breed_id']].iloc[train_rnd_ids].values,
                           size), \
                   DatasetCatDog([cashe_imgs[i] for i in test_rnd_ids],
                           df[['name', 'breed_id']].iloc[test_rnd_ids].values,
                           size))


def resize_images_tf(images, size):
    with tf.Session() as sess:
        image_ph = tf.placeholder(tf.float32, (None, None, None, images.shape[-1]))

        img = tf.image.resize_bilinear(image_ph, size, align_corners=True)

        return sess.run(img, feed_dict={image_ph: images})


class Dataset:
    def __init__(self, *data):
        self.data = data
        self.data_aug = self.data[0].copy()

        self.k = 2
        self.cut_size = len(self.data[0])*2
        self.buf_arr = np.arange(self.cut_size)

        self.batch_size = 1
        self.indexes = []
        self.repeat_times = -1
        self.image_ph = None
        self.img = None
        self._aug_flag = None
        self._sfl_flag = None

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

    def batch(self, batch_size):
        self.batch_size = batch_size
        self.cut_size = len(self.data[0])*self.k // self.batch_size

        return self

    def repeat(self, times=-1):
        self.repeat_times = times

        return self

    def augment(self):
        if self._aug_flag is None:
            self._aug_flag = self.repeat_times

        # Implement your augmentation func here

        return self

    def get_batch(self):
        counter = 0

        while True:
            if counter == self.repeat_times:
                break

            if counter != 0:
                # Augment data in the cache for each 'self.repeat' iteration if self.augment()
                # was called after self.repeat()
                if self.repeat_times == self._aug_flag:
                    self.augment()

                # Shuffle data in the cache for each 'self.repeat' iteration if self.shuffle()
                # was called after self.repeat()
                if self.repeat_times == self._sfl_flag:
                    self.shuffle()

            if self._aug_flag is None:
                self.indexes = self.buf_arr[:self.cut_size // self.k * self.batch_size].reshape(
                    [-1, self.batch_size])

                for idxs in self.indexes:
                    yield (arr[idxs] for arr in self.data)
            else:
                self.indexes = self.buf_arr[:self.cut_size * self.batch_size].reshape(
                    [-1, self.batch_size])

                for idxs in self.indexes:
                    yield (np.concatenate([self.data[0], self.data_aug], axis=0)[idxs],
                           *(np.tile(arr, self.k)[idxs] for arr in self.data[1:]))

            counter += 1

        # Reset all the properties
        self._aug_flag, self._sfl_flag, self.repeat_times = None, None, -1

    # TODO: Finish these three methods
    def get_data(self):
        """Gets the whole dataset.

        Returns:
            Tuple of Numpy arrays: `(img_aug, species, breeds)`.
        """

        return self.img_aug, self.labels

    def save(self, path):
        # Reshape into tf.image.encode_jpeg format
        img_ph = tf.placeholder(tf.uint8, (None, None, None, self.imgs_cashe.shape[-1]))
        filename_ph = tf.placeholder(tf.uint8, (None, None, None, self.imgs_cashe.shape[-1]))
        # images = tf.reshape(tf.cast(img_ph, tf.uint8), [16, 16, 1])

        # Resize
        images_resized = tf.image.resize_images(img_ph, self.size)

        # Encode
        images_encoded = tf.image.encode_jpeg(img_ph)

        # Create a files name
        fname = tf.constant(path + filename_ph + ".jpeg")

        # Write files
        fwrite = tf.write_file(fname, images_encoded)


        with tf.Session() as sess:

            feed_dict_train = {img_ph: self.img_aug, filename_ph: y_true_batch}

            sess.run(fwrite, feed_dict=feed_dict_train)

        print("Images Saved!")

    def load(self, path, extension='jpeg'):
        import glob

        imagepaths = glob.glob(path + '/*.{}'.format(extension), recursive=True)
        input(imagepaths)

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

        cashe_imgs = np.array([*generator()]) if resize else [*generator()]


if __name__ == "__main__":
    # TODO: Argpars
    folder = 'dataset/'
    save_to = 'dataset/'
    new_size = (128, 128)

    import time

    start = time.time()
    for a, b in load_data(folder, new_size, resize=True, channels=1, mode='dog').batch(32).shuffle().repeat(5).augment().get_batch():
        pass
        # for img in a:
        #     plt.imshow(np.squeeze(img), cmap='gray')
        #     plt.colorbar()
        #     plt.show()

    print(time.time() - start)
    input(123)
