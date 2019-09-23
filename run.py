import argparse

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from utils import get_config
from models import GAN_PG, plot
from dataset import preprocess_dataset


def generate(path, stage_num=None):

    model = GAN_PG(**config)

    model(stage_num or -1, training=False)
    model.load_weights(path)

    results = model.generator([noise, labels], training=False)
    results = results if stage_num is None else [results[stage_num]]
    for img in results:
        res = model.sess.run(img)
        plot(res, y_train.shape[-1], 10, title='results_{}'.format(stage_num or len(results)))

        plt.savefig('{}/results/{}_{}.png'.format(config['folder'], stage_num, config['gan_mode']), bbox_inches='tight')

        if show:
            plt.show()


if __name__ == "__main__":
    size = 10
    show = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', type=str, default='./config.py', help='a path to the config.py')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='a dataset which the model will be trained on')

    args = parser.parse_args()

    # load config
    config = get_config(args.path_config)

    #TODO: Remove
    # Load the dataset
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

    generate('{}/weights'.format(config['folder']))

    print('Generation done!')
