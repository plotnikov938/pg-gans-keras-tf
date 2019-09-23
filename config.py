import tensorflow as tf


# Settings
def load():
    """A function returns dict with the configured parameters"""

    train_stages = [
        {'size': (4, 4, 128), 'train_epochs': 50, 'num': 0},
        {'size': (8, 8, 64), 'train_epochs': 150, 'num': 1},
        {'size': (16, 16, 32), 'train_epochs': 200, 'num': 2},
        {'size': (32, 32, 16), 'train_epochs': 300, 'num': 3},
    ]

    use_tpu = False
    dataset = 'cifar10'  # 'mnist'
    folder = '.'
    restore = True  # Restore a pre-trained model, if one exists
    sess_config = None  # Configure here tensorflow session if necessary

    gan_mode = 'ra-gan'  # See `adversarials.py` for more information
    conditional = True
    latent_size = 512
    use_wscale = True
    gen_act = tf.keras.layers.LeakyReLU(0.2)
    dis_act = tf.keras.layers.LeakyReLU(0.2)
    batch_norm = False
    pixel_norm = True
    drop_rate = 0
    mixup = 0.0
    batch_size = 128
    learning_rate = 0.0002
    dis_train_iters = 1  # Iterations to train the discriminator
    gen_train_iters = 1  # Iterations to train the generator

    buffer_size = 30000  # How many generated samples would you like to store in the buffer?
    buffer_epoch_depth = 30  # From how many previous epochs samples should be stored in the buffer?

    return locals()
