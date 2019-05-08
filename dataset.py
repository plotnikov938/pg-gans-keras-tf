import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import glob
# from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import tensorflow as tf


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


# def thumbnail(img, size, resample=Image.BICUBIC):
#     # Get new size of the image preserving the aspect ratio
#     x, y = old_size = img.size
#     if y < x:
#         y = int(max(y * size[0] / x, 1))
#         x = int(size[0])
#     else:
#         x = int(max(x * size[1] / y, 1))
#         y = int(size[1])
#     size = x, y
#
#     if size != old_size:
#         img = img.resize(size, resample)
#
#     return img


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
    img = Image.open(path_to_imgs + filename)

    # make_square(borders)
    cropped_img = img.crop(borders)

    converted_img = cropped_img.convert(convert)
    converted_img = thumbnail(converted_img, size, Image.ANTIALIAS)

    return converted_img

    # plt.imshow(pix, cmap='gist_gray', clim=(0, 255))
    # plt.colorbar()
    # plt.show()


def image_preprocessing_tf(filename, borders, size, convert="L"):
    img = Image.open(path_to_imgs + filename)

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
          path_to_ants: path where annotations are stored

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
        :param size:
        :param crop_frac: (float)
            The lower bound of cropping scale to randomly decreasing size of the source image
        :param brightness_delta:
        :param contrast_min_max:
        :param stddev_max:
        :return:
            Resized images. Data range from 0.0 to 1.0. [batch_size, size[0], size[1], channels]
        """

        # Random cropping with shifting
        frac = tf.random_uniform([], minval=crop_frac)
        img_shapes = tf.shape(img)
        img_shapes_squared = tf.cast(tf.minimum(img_shapes[1:], tf.reduce_min(img_shapes[1:3])), tf.float32)
        crop_size = tf.cast(img_shapes_squared*frac, dtype=tf.int32)
        try:
            img = tf.image.random_crop(img, (img_shapes[0], crop_size[0], crop_size[1], img_shapes[-1]))
        except ValueError:
            img_shapes_squared = tf.cast(tf.minimum(img_shapes, tf.reduce_min(img_shapes[:2])), tf.float32)
            crop_size = tf.cast(img_shapes_squared * frac, dtype=tf.int32)
            img = tf.image.random_crop(img, (crop_size[0], crop_size[1], img_shapes[-1]))

        # # Random flipping
        if flip_h:
            img = tf.image.random_flip_left_right(img)
        if flip_v:
            # TODO: Прочекать, может быть не совсем полезная фича с точки зрения репрезентативности данных
            img = tf.image.random_flip_up_down(img)

        # Random rotation
        if rot90:
            img = tf.image.rot90(img, tf.random_uniform(shape=[], minval=0, maxval=3, dtype=tf.int32))

        # TODO: for discrim train set range from -1 to 1
        img = tf.image.random_contrast(img, *contrast_min_max)

        # Gaussian Noise
        stddev = tf.random_uniform(shape=(), maxval=stddev_max)
        noise = tf.random_normal(shape=tf.shape(img), stddev=stddev)
        img += noise

        # Random_brightness
        delta = tf.random_uniform(shape=(), minval=-brightness_delta, maxval=brightness_delta)
        img = tf.clip_by_value(img - delta, 0.0, 1.0)

        return tf.image.resize(img, size)


def load_data(folder, size, mask=None, channels=1, scale=.3):
        """
        :param channels: (int)
            'cat': Loads only images of cats
            'dog': Loads only images of dogs
            None: Loads all images
        :param channels: (int)
            The number of channels to execute from source image
        :param scale: (float)
            The value is used to scale up boundary box of an animal's head on an image
        """

        df = load_annotations(folder + 'annotations')
        if df is None:
            df = parse_annotations(path_to_ants)
            save_annotations(folder + 'annotations', df)

        # Mask the dataset
        if mask is not None:
            df = df[df["name"] == mask]

        # Turn specie name into '0-1' label
        df["name"] = df["name"] == 'cat'

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

        # Convert imagepaths to Tensor
        imagepaths = (path_to_imgs + array[:, 0]).astype('str')
        imagepaths = tf.constant(imagepaths)

        def _func(imagepath, borders):
            image_raw = tf.read_file(imagepath)
            image = tf.image.decode_and_crop_jpeg(image_raw, borders, channels)
            # image = tf.image.resize(image, size)
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

        # cashe_imgs = np.array([*generator()])
        cashe_imgs = [*generator()]

        return Dataset(cashe_imgs, df[['name', 'breed_id']].values, size)


class Dataset:
    def __init__(self, imgs_cashe, labels, size):
        self.imgs_cashe = imgs_cashe
        self.labels = labels.astype(np.int32)
        self.size = size

        self.cut_size = len(self.imgs_cashe)
        self.buf_arr = np.arange(self.cut_size)

        self.img_aug = np.zeros((self.cut_size, *size, imgs_cashe[0].shape[-1]))

        self.batch_size = 1
        self.indexes = []
        self.repeat_times = None
        self.image_ph = None
        self.img = None
        self.aug = None


    def augment(self, renew=False, **kwargs):
        # Used to determine the order of methods calling
        self.aug = self.repeat_times

        if renew:
            tf.reset_default_graph()

        if self.image_ph is None or renew:

            self.image_ph = tf.placeholder(tf.float32, (None, None, None, 1), name="x")

            self.image = \
                augmentation(self.image_ph, size=self.size, **kwargs)

        with tf.Session() as sess:
            self.img_aug = sess.run(self.image, feed_dict={self.image_ph: self.imgs_cashe})

        return self

    def augment2(self, renew=False, **kwargs):
        # Used to determine the order of methods calling
        self.aug = self.repeat_times

        if renew:
            tf.reset_default_graph()

        if self.image_ph is None or renew:

            self.image_ph = tf.placeholder(tf.float32, (None, None, 1), name="x")

            self.image = \
                augmentation(self.image_ph, size=self.size, **kwargs)

        with tf.Session() as sess:
            for idx, image in enumerate(self.imgs_cashe):
                self.img_aug[idx] = sess.run(self.image, feed_dict={self.image_ph: image})

        return self

    def save(self, path):
        pass

    def shuffle(self):
        np.random.shuffle(self.buf_arr)

        return self

    def batch(self, batch_size):
        self.batch_size = batch_size
        self.cut_size = len(self.imgs_cashe) // self.batch_size

        return self

    def repeat(self, times=-1):
        self.repeat_times = times

        return self

    def get_batch(self):
        counter = 0

        while True:
            if counter == self.repeat_times:
                break

            self.indexes = self.buf_arr[-self.cut_size * self.batch_size:].reshape(
                [-1, self.batch_size])[::-1]

            # Augmenting images in the cache for each 'self.repeat' iteration if self.augment()
            # was called after self.repeat()
            if self.repeat_times == self.aug:
                self.augment2()

            for j in range(self.indexes.shape[0]):
                indexes = self.indexes[j]

                # yield [self.img_aug[i] for i in indexes], self.labels[indexes]
                yield self.img_aug[indexes], self.labels[indexes]

            counter += 1

        self.aug = None

    def get_data(self):
        """Gets the whole dataset.

        Returns:
            Tuple of Numpy arrays: `(img_aug, species, breeds)`.
        """

        return self.img_aug, self.labels


if __name__ == "__main__":
    # TODO: Argpars
    folder = 'dataset/'
    save_to = 'dataset/'
    path_to_imgs = folder + 'images/'
    path_to_ants = folder + 'annotations/'
    new_size = (128, 128)

    import time

    start = time.time()
    for a, b in load_data(folder, new_size, channels=1, mask='dog').batch(32).shuffle().repeat(5).augment2().get_batch():
        pass
        # for img in a:
        #     plt.imshow(np.squeeze(img), cmap='gray')
        #     plt.colorbar()
        #     plt.show()

    print(time.time() - start)
    input(123)
