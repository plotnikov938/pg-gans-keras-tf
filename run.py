import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from utils import get_config
from models import GAN_PG, plot
from dataset import preprocess_dataset


def generate(path, stage_nums=None):
    stage_nums = [stage_nums] if not isinstance(stage_nums, (tuple, list)) else stage_nums

    tf.keras.backend.clear_session()

    model = GAN_PG(**config)

    model.compile_model()

    model.sess.run(tf.global_variables_initializer())

    # Restore the weights
    model.load_weights(path)

    inputs = [noise, labels] if config["conditional"] is True else noise

    for stage in stage_nums:
        generated_img = model.generate(inputs, stage)

        plot(generated_img, y_train.shape[-1], labels.shape[-1], title='stage_{}'.format(stage))

        plt.savefig('{}/results/{}_{}.png'.format(config['folder'], stage, config['gan_mode']), bbox_inches='tight')

        if args.show:
            plt.show()


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--path_config', type=str, default='./config.py', help='a path to the config.py')
    # parser.add_argument('-f', '--filename', type=str, default='{}_{}.png', help='a filename to save generated images')
    parser.add_argument('-s', '--show', action='store_true', help='show generated images')
    parser.add_argument('-l', '--label', type=int, default=-1, choices=range(-2, 10, 1),
                        help='Use this option in case the model was trained conditionally: '
                             '\n   [0-9] - generate images with selected label '
                             '\n   -1 (default) - generate images for each label presented '
                             '\n   -2 - generate randomly labeled images')
    parser.add_argument('--stage_nums', type=int, nargs='*', default=-1,
                        help='defines the stages for which images will be generated')

    if len(sys.argv) == 2:
        if sys.argv[-1] in ['-h', '--help']:
            parser.print_help(sys.stderr)
            sys.exit(1)

    args = parser.parse_args('-s --stage_nums -2 -1 -l -2'.split(' '))

    # load config
    config = get_config(args.path_config)

    if config['dataset'] is 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif config['dataset'] is 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Load your custom dataset here
    else:
        raise NotImplementedError("Load your custom dataset here")

    X_train, y_train = preprocess_dataset((X_train, y_train), (X_test, y_test))
    config['channels'] = X_train.shape[-1]

    size = 10
    noise = np.random.normal(size=(y_train.shape[-1] * size, config['latent_size'])).astype(np.float32)
    # Preprocess labels
    if args.label == -1:
        labels = np.tile(np.arange(y_train.shape[-1])[:, None], (1, size)).reshape(-1)
    elif args.label == -2:
        labels = np.random.randint(0, 9, size=y_train.shape[-1]*size)
    else:
        labels = np.ones((y_train.shape[-1]*size))*args.label
    labels = tf.keras.utils.to_categorical(labels, y_train.shape[-1]).astype(np.float32)

    config['labels_emb_size'] = y_train.shape[-1] if config['conditional'] else None

    generate('{}/weights'.format(config['folder']), args.stage_nums)

    print('Generation done!')
