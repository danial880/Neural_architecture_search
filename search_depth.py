import os
import sys
import glob
import torch
import utils
import logging
import numpy as np
from time import time
from time import strftime
from trainer import Train
from torch.backends import cudnn
from model_search import ModelSearch
from network_mix import NetworkMixArch


def setup_start(config):
    exp_name = config['logging']['exp_name']
    txt_name = config['logging']['txt_name']
    save_name = 'Search-{}-{}'.format(exp_name, strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(save_name, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_name, txt_name))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return save_name

def main(save_name):
    if not torch.cuda.is_available():
        logging.info('GPU not available.')
        sys.exit(1)
    config = utils.load_yaml()
    device = config['device']['gpu']
    seed = config['hyperparameters']['seed']
    np.random.seed(seed)
    torch.cuda.set_device(device)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)
    logging.info("dataset_to_run = %s", config['dataset_to_run'])
    logging.info('GPU Device = %d' % device)
    logging.info("Hyperparameters = %s", config['hyperparameters'])
    logging.info("Search Parameters = %s", config['search_parameters'])
    config['flags']['resume'] = False
    trainer = Train(config, save_name)
    # Log sub dataset under consideration.
    class_labels = trainer.get_desired_dataset(logging)
    search_model = ModelSearch(config, class_labels)
    search_model.search_depth(class_labels, trainer.train_test, trainer.train)


if __name__ == '__main__':
    start_time = time()
    config = utils.load_yaml()
    save_name = setup_start(config)
    main(save_name)
    end_time = time()
    duration = end_time - start_time
    logging.info('Total Search Time: %ds', duration)
