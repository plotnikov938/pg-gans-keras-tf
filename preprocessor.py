import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import glob
from PIL import Image, ImageFilter, ImageDraw


fields_border = ['xmin', 'ymin', 'xmax', 'ymax']
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


def thumbnail(img, size, resample=Image.BICUBIC):
    # Get new size of the image preserving the aspect ratio
    x, y = old_size = img.size
    if y < x:
        y = int(max(y * size[0] / x, 1))
        x = int(size[0])
    else:
        x = int(max(x * size[1] / y, 1))
        y = int(size[1])
    size = x, y

    if size != old_size:
        img = img.resize(size, resample)

    return img


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


def image_augmentation():
    pass


def image_preprocessing(filename, borders, size, convert="L"):
    img = Image.open(path_to_imgs + filename)

    make_square(borders)
    cropped_img = img.crop(borders)

    converted_img = cropped_img.convert(convert)
    converted_img = thumbnail(converted_img, size, Image.ANTIALIAS)

    import matplotlib.pyplot as plt
    pix = np.array(converted_img)

    plt.imshow(pix, cmap='gist_gray', clim=(0, 255))
    plt.colorbar()
    plt.show()


def breed2id(breeds):
    idxs = np.lexsort(breeds[None, :], axis=0)
    b = breeds[idxs]
    args = np.argwhere(b[1:] != b[:-1]).ravel() + 1

    for idx, split in enumerate(np.split(b, args)):
        split[:] = idx

    breeds[idxs] = b

    return breeds


if __name__ == "__main__":
    # TODO: Argpars
    path_to_imgs = 'images/'
    path_to_ants = 'annotations/'
    save_to = 'dataset/'
    new_size = (24, 24)

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

    # Cast coords of head from str to int
    df[fields_border] = df[fields_border].astype('int')

    # Create breed_id separately for cats and dogs for a classification problem
    df['breed_id'] = None

    slice_dogs = df['name'] == 'dog'
    dog_breeds = df.loc[slice_dogs, 'breed'].values
    df.loc[slice_dogs, 'breed_id'] = breed2id(dog_breeds)

    slice_cats = df['name'] == 'cat'
    cat_breeds = df.loc[slice_cats, 'breed'].values
    df.loc[slice_cats, 'breed_id'] = breed2id(cat_breeds)

    # Pre-process the images
    for row in df[['filename'] + fields_border].values:
        filename, borders = row[0], row[1:]
        image_preprocessing(filename, borders, new_size, convert="L")

    # TODO: Convert imgs to numpy array and save as int array (or csv?)
    # TODO: Data_augmentation on the fly (during training?)
