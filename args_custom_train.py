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

"""
Example: python args_custom_train.py --seed 21 --batch_size 32 --learning_rate 0.01 --epochs 6  --report_freq 2 --gpu 0 --dataset_to_run CIFAR10 

"""
def get_custom_model(cfg, name):
    logging.info("dataset_to_run = %s", cfg['dataset_to_run'])
    logging.info('GPU Device = %d' % cfg['device']['gpu'])
    logging.info("Hyperparameters = %s", cfg['hyperparameters'])
    logging.info("Search Parameters = %s", cfg['custom_network'])
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
    # get data iterators  
    trainer = Train(cfg, name)
    class_labels = trainer.get_desired_dataset(logging)
    # Log sub dataset under consideration.
    # intialize model
    input_shape = utils.get_input_shape(cfg['dataset_to_run'])
    grayscale = utils.get_grayscale(config['dataset_to_run'])
    channels = cfg['custom_network']['channels']
    layers = cfg['custom_network']['layers']
    ops = cfg['custom_network']['ops']
    kernels = cfg['custom_network']['kernels']
    logging.info("Model Channels (width) %s \nModel Layers (depth) %s",
                 channels, layers)
    logging.info("Model Ops %s \nModel Kernels %s", ops, kernels)
    model = NetworkMixArch(channels, len(class_labels), layers, ops, kernels,
                           input_shape, grayscale)
    model = model.cuda()
    return model, trainer
    

def train_custom(cfg, name):
    model, trainer = get_custom_model(cfg, name)
    # start training
    train_acc, test_acc = trainer.train_test(model)
    print('Training Acc = {}\nValidation Acc = {}\n'.format(train_acc, test_acc))


def update_config_with_args(config, args):
    # hyperparameters
    config['hyperparameters']['seed'] = args.seed
    config['hyperparameters']['valid_size'] = args.valid_size
    config['hyperparameters']['batch_size'] = args.batch_size
    config['hyperparameters']['learning_rate'] = args.learning_rate
    config['hyperparameters']['weight_decay'] = args.weight_decay
    config['hyperparameters']['momentum'] = args.momentum
    config['hyperparameters']['grad_clip'] = args.grad_clip
    config['hyperparameters']['epochs'] = args.epochs
    config['hyperparameters']['cutout_length'] = args.cutout_length
    # flags
    config['flags']['auto_augment'] = args.auto_augment
    config['flags']['cutout'] = args.cutout
    config['flags']['resume'] = args.resume
    config['flags']['save'] = args.save
    # logging
    config['logging']['report_freq'] = args.report_freq
    # device
    config['device']['gpu'] = args.gpu
    # dataset to run
    config['dataset_to_run'] = args.dataset_to_run
    # paths
    config['paths']['data_dir'] = args.data_dir
    config['paths']['resume_checkpoint'] = args.resume_checkpoint
    # custom network
    config['custom_network']['channels'] = args.channels
    config['custom_network']['layers'] = args.layers
    config['custom_network']['ops'] = args.ops
    config['custom_network']['kernels'] = args.kernels
    # return updated config
    return config


def init_config_logs(args):
    config = utils.load_yaml()
    exp_name = args.dataset_to_run
    txt_name = exp_name + '.txt'
    save_name = 'Train-{}-{}'.format(exp_name, strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(save_name, scripts_to_save=glob.glob('*.py'))
    # logging
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_name, txt_name))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return config, save_name


def get_args():
    parser = argparse.ArgumentParser(description='Neural Network Configuration')
    # hyperparameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--valid_size', type=float, default=0,
                        help='Validation set size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=str, default='3e-4',
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--grad_clip', type=float, default=5, 
                        help='Gradient clipping')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--cutout_length', type=int, default=16,
                        help='Cutout length')
    # custom_network
    parser.add_argument('--channels', type=int, default=48,
                        help='Number of channels')
    parser.add_argument('--layers', type=int, default=6,
                        help='Number of layers')
    parser.add_argument('--ops', nargs='+', type=int,
                        default=[0, 1, 0, 1, 0, 1], help='List of operations')
    parser.add_argument('--kernels', nargs='+', type=int,
                        default=[3, 3, 5, 5, 7, 7], help='List of kernel sizes')
    # flags
    parser.add_argument('--auto_augment', type=bool, default=False,
                        help='Enable auto augmentation')
    parser.add_argument('--cutout', type=bool, default=True,
                        help='Enable cutout augmentation')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume training from checkpoint')
    parser.add_argument('--save', type=bool, default=True,
                        help='Save the trained model')
    # logging
    parser.add_argument('--report_freq', type=int, default=1,
                        help='Reporting frequency')
    # device
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    # dataset to run
    parser.add_argument('--dataset_to_run', type=str, default='DTD',
                        help='Dataset to run')
    # paths
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for data')
    parser.add_argument('--resume_checkpoint', type=str, default='./',
                        help='Path to resume checkpoint')
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = get_args()
    config, save_name = init_config_logs(args)
    updated_config = update_config_with_args(config, args)
    start_time = time.time() 
    # run training
    train_custom(updated_config, save_name)
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total Search Time: %ds', duration)

