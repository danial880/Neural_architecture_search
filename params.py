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
import torch.nn as nn
import torch.backends.cudnn as cudnn
from time import strftime
from trainer import Train
from random import randrange
from network_mix import NetworkMixArch
from utils import count_parameters_in_MB
from train_custom_net import get_custom_model

def get_params(cfg, name):
    model, trainer = get_custom_model(cfg, name)
    count = count_parameters_in_MB(model)
    logging.info("Parameter Count (MB) =  {}\n".format(count))
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # start training 
    start_time = time.time()
    train_acc, test_acc = trainer.infer(trainer.test_queue, model, criterion)
    end_time = time.time() - start_time
    num_images = len(trainer.test_queue)
    logging.info("Latency =  {:.4f} seconds\n".format(end_time/num_images))


if __name__ == "__main__":
    # set logging
    config = utils.load_yaml()
    exp_name = config['logging']['exp_name']
    txt_name = config['logging']['txt_name']
    save_name = ''
    log_format = '%(message)s'
    logging.basicConfig(filename='params.txt', level=logging.INFO,
                        format=log_format, filemode='a')
    get_params(config, save_name)
