import argparse

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
    model.load_weights(path, tpu=config['use_tpu'])

    for stage in stage_nums:

        generated_img = model.generate([noise, labels], stage)

        plot(generated_img, y_train.shape[-1], labels.shape[-1], title='stage_{}'.format(stage))

        plt.savefig('{}/results/{}_{}.png'.format(config['folder'], stage, config['gan_mode']), bbox_inches='tight')

        if args.show:
            plt.show()


if __name__ == "__main__":
    size = 10

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', type=str, default='./config.py', help='a path to the config.py')
    parser.add_argument('--show', '-s', action='store_true', help='show generated images')
    parser.add_argument('--stage_nums', type=int, nargs='*', default=-1,
                        help='defines the stages for which images will be generated')

    args = parser.parse_args('-s --stage_nums -2 -1'.split(' '))

    # load config
    config = get_config(args.path_config)

    #TODO: Remove
    if config['dataset'] is 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif config['dataset'] is 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Load your custom dataset here
    else:
        raise NotImplementedError("Load your custom dataset here")

    X_train, y_train = preprocess_dataset((X_train, y_train), (X_test, y_test))
    config['channels'] = X_train.shape[-1]

    noise = np.random.normal(size=(y_train.shape[-1] * size, config['latent_size'])).astype(np.float32)
    labels = np.tile(np.arange(y_train.shape[-1])[:, None], (1, size)).reshape(-1)
    labels = tf.keras.utils.to_categorical(labels, y_train.shape[-1]).astype(np.float32)

    config['labels_emb_size'] = y_train.shape[-1] if config['conditional'] else None

    generate('{}/weights'.format(config['folder']), args.stage_nums)

    print('Generation done!')
