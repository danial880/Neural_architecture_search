import utils
import random
import logging
import numpy as np
from random import randrange


def get_random_model(cfg):
    """Generate random network models based on the given configuration.

    Args:
        cfg (dict): Configuration dictionary containing parameters for random
                    network generation.

    Returns:
        None
    """
    for i in range(cfg['num_net']):
        kernels_list = [3,5,7]
        channels = randrange(cfg['min_width_channels'],
                             cfg['max_width_channels']+cfg['width_step'],
                             cfg['width_step'])
        layers = randrange(cfg['min_depth_layers'],
                           cfg['max_depth_layers']+cfg['depth_step'],
                           cfg['depth_step'])
        ops = np.array([random.randint(0,1) for _ in range(layers)])
        kernels = np.array([random.choice(kernels_list) for _ in range(layers)])
        network = {'channels': channels,
                   'layers' : layers,
                   'ops' : list(ops),
                   'kernels' : list(kernels)}
        logging.info("Network {}\n{}\n".format(i+1,network))


if __name__ == "__main__":
    config = utils.load_yaml()
    config = config['random_network_gen']
    # set logging
    log_format = '%(message)s'
    logging.basicConfig(filename='random_network.txt', level=logging.INFO,
                        format=log_format, filemode='w')
    get_random_model(config)
