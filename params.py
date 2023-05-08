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
from utils import count_parameters_in_MB

def get_params(cfg, name):
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
    count = count_parameters_in_MB(model)
    logging.info("Parameter Count (MB) =  {}\n".format(count))
    model = model.cuda()
    # start training
    cfg['dataset_to_run'] = 'CIFAR10'
    cfg['hyperparameters']['input_shape'] = 32
    cfg['hyperparameters']['epochs'] = 1
    cfg['flags']['save'] = False
    trainer = Train(cfg, name)
    # Log sub dataset under consideration.
    class_labels = trainer.get_desired_dataset(logging)

    # start training
    
    start_time = time.time()
    train_acc, test_acc = trainer.train_test(model)
    end_time = time.time() - start_time
    logging.info("Latency =  {:.2f} seconds\n".format(end_time))


if __name__ == "__main__":

    # set logging
    config = utils.load_yaml()
    exp_name = config['logging']['exp_name']
    txt_name = config['logging']['txt_name']
    save_name = ''

    log_format = '%(message)s'
    logging.basicConfig(filename='params.txt', level=logging.INFO, format=log_format, filemode='a')
    get_params(config, save_name)
