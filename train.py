import os
import sys
import glob
import argparse
import warnings

import numpy as np
import tensorflow as tf

from utils import Scheduler, make_gif, empty_context_mgr, get_config
from dataset import preprocess_dataset
from models import GAN_PG


def train():
    tf.keras.backend.clear_session()
    with strategy.scope() if config['use_tpu'] else empty_context_mgr():

        model = GAN_PG(**config)

        # Define optimizers
        optimizer_g = tf.train.AdamOptimizer(learning_rate=config['learning_rate'], beta1=0.0)
        optimizer_d = tf.train.AdamOptimizer(learning_rate=config['learning_rate'], beta1=0.0)

        # Compile the model
        model.compile_model(optimizer_g=optimizer_g, optimizer_d=optimizer_d, loss=config['gan_mode'],
                            tpu_strategy=strategy, resolver=resolver, config=config['sess_config'])

        if config['restore']:
            model.load_weights('{}/weights'.format(config['folder']))

        # Prepare inputs
        inputs = (X_train, y_train) if config['conditional'] else X_train

        # Train
        for stage in config['train_stages']:
            # Get training stage num
            stage_num = config['train_stages'].index(stage)

            print('\nProcessing stage: {}  with image size {} =========================================='.format(
                stage_num, stage['size']))

            # Define schedulers
            alpha_scheduler = Scheduler(stage['train_epochs'], [0, 0.5], [0, 1.0])
            learning_rate_scheduler = Scheduler(stage['train_epochs'], [0, 0.5], [stage['lr']*0.1, stage['lr']])

            model.fit_stage(inputs, config['batch_size'], stage_num=stage_num,
                            alpha_scheduler=alpha_scheduler,
                            learning_rate_scheduler=learning_rate_scheduler,
                            folder=config['folder'], save_epoch=config['save_epoch'],
                            seed_noise=seed_noise, seed_labels=seed_labels
                            )

    make_gif(glob.iglob('{}/progress/*.png'.format(config['folder'])),
             '{}/progress/{}_{}.gif'.format(config['folder'], config['gan_mode'], 'progress'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', type=str, default='./config.py', help='a path to the config.py')

    if len(sys.argv) == 2:
        if sys.argv[-1] in ['-h', '--help']:
            parser.print_help(sys.stderr)
            sys.exit(1)

    args = parser.parse_args()

    # load config
    config = get_config(args.path_config)

    # Create the folder to save the training progress
    try:
        os.makedirs(config['folder'])
    except FileExistsError:
        pass
    try:
        os.makedirs(config['folder'] + '/progress')
    except FileExistsError:
        pass

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

    labels_emb_size = y_train.shape[-1] if config['conditional'] else None

    # Set seed parameters to control the training process
    seed_noise = np.random.normal(size=(y_train.shape[-1] * 10, config['latent_size'])).astype(np.float32)
    seed_labels = np.tile(np.arange(y_train.shape[-1])[:, None], (1, 10)).reshape(-1)
    seed_labels = tf.keras.utils.to_categorical(seed_labels, y_train.shape[-1]).astype(np.float32)

    if config['use_tpu']:
        if config['buffer_size']:
            warnings.warn('The using of the buffer on TPUs is not implemented')
            config['buffer_size'] = 0

        assert os.environ[
            'COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'
        assert float(
            '.'.join(tf.__version__.split('.')[:2])) >= 1.14, 'Make sure that Tensorflow version is at least 1.14'

        TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu=TPU_WORKER)
        tf.contrib.distribute.initialize_tpu_system(resolver)
        strategy = tf.contrib.distribute.TPUStrategy(resolver)
    else:
        resolver, strategy = None, None

    # Run training
    train()
