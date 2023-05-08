import os
import sys
import glob
import time
import torch
import utils
import random
import logging
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from time import strftime
from trainer import Train
from random import randrange
from network_mix import NetworkMixArch



def get_random_model(cfg):
    # set gpu device
    torch.cuda.set_device(cfg['device']['gpu'])
    # set cudnn flags
    cudnn.enabled = True
    cudnn.benchmark = True
    # set seed
    seed = cfg['hyperparameters']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # get class labels
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    cifar_classes = len(np.array(class_labels))
    total_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                     'frog', 'horse', 'ship', 'truck']
    classes = []
    for i in np.array(class_labels):
        classes.append(total_classes[i])
    # intialize model
    channels = cfg['custom_network']['channels']
    layers = cfg['custom_network']['layers']
    ops = cfg['custom_network']['ops']
    kernels = cfg['custom_network']['kernels']
    logging.info("Model Channels (width) %s \nModel Layers (depth) %s",
                 channels, layers)
    logging.info("Model Ops %s \nModel Kernels %s", ops, kernels)
    model = NetworkMixArch(channels, cifar_classes, layers, ops, kernels, 32)
    model = model.cuda()
    return model, classes

def train_random(cfg, name):
    cfg['dataset_to_run'] = 'CIFAR10'
    cfg['hyperparameters']['input_shape'] = 32
    logging.info("dataset_to_run = %s", cfg['dataset_to_run'])
    logging.info('GPU Device = %d' % cfg['device']['gpu'])
    logging.info("Hyperparameters = %s", cfg['hyperparameters'])
    logging.info("Search Parameters = %s", cfg['custom_network'])
    # get random model
    model, classes = get_random_model(cfg)
    # get data iterators
    
    trainer = Train(cfg, name)
    # Log sub dataset under consideration.
    class_labels = trainer.get_desired_dataset(logging)

    # start training
    train_acc, test_acc = trainer.train_test(model)
    print('Training Acc = {}\nValidation Acc = {}\n'.format(train_acc, test_acc))


if __name__ == "__main__":
    # create save directory
    config = utils.load_yaml()
    exp_name = config['logging']['exp_name']
    txt_name = config['logging']['txt_name']
    save_name = 'Search-{}-{}'.format(exp_name, strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(save_name, scripts_to_save=glob.glob('*.py'))
    # logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_name, txt_name))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    start_time = time.time() 
    # run training
    train_random(config, save_name)
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total Search Time: %ds', duration)
