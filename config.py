import tensorflow as tf


# Settings
def load():
    """A function returns dict with the configured parameters"""

    train_stages = [
        {'size': (4, 4, 256), 'train_epochs': 50, 'num': 0, 'lr': 0.0002},
        {'size': (8, 8, 256), 'train_epochs': 150, 'num': 1, 'lr': 0.0002},
        {'size': (16, 16, 256), 'train_epochs': 200, 'num': 2, 'lr': 0.0002},
        {'size': (32, 32, 128), 'train_epochs': 300, 'num': 3, 'lr': 0.0001},
    ]

    use_tpu = True
    dataset = 'cifar10'  # 'mnist'
    folder = '.'
    restore = False  # Restore a pre-trained model, if one exists
    sess_config = None  # Configure here tensorflow session if necessary

    gan_mode = 'ra-gan'  # See `adversarials.py` for more information
    conditional = True
    latent_size = 256
    use_wscale = False
    gen_act = tf.keras.layers.LeakyReLU(0.2)
    dis_act = tf.keras.layers.LeakyReLU(0.2)
    batch_norm = False
    pixel_norm = True
    drop_rate = 0
    mixup = 0.0

    buffer_size = 30000  # How many generated samples would you like to store in the buffer?
    buffer_epoch_depth = 30  # From how many previous epochs samples should be stored in the buffer?

    batch_size = 128
    learning_rate = 0.0002
    dis_train_iters = 1  # Iterations to train the discriminator
    gen_train_iters = 1  # Iterations to train the generator
    save_epoch = 5  # Save weights, losses plot and generated images every N epochs

    return locals()
