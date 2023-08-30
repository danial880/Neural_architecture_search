import os
import sys
import glob
import torch
import utils
import logging
import argparse
import numpy as np
from time import time
from time import strftime
from trainer import Train
from torch.backends import cudnn
from model_search import ModelSearch
from network_mix import NetworkMixArch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


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
    # search parameters
    config['search_parameters']['target_acc'] = args.target_acc
    config['search_parameters']['target_acc_tolerance'] = args.target_acc_tolerance
    config['search_parameters']['ch_drop_tolerance'] = args.ch_drop_tolerance
    config['search_parameters']['min_width_channels'] = args.min_width_channels
    config['search_parameters']['max_width_channels'] = args.max_width_channels
    config['search_parameters']['width_resolution'] = args.width_resolution
    config['search_parameters']['min_depth_layers'] = args.min_depth_layers
    config['search_parameters']['max_depth_layers'] = args.max_depth_layers
    # return updated config
    return config


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
    # Search parameters
    parser.add_argument('--target_acc', type=float, default=99,
                        help='Target accuracy')
    parser.add_argument('--target_acc_tolerance', type=float, default=0.5,
                        help='Target accuracy tolerance')
    parser.add_argument('--ch_drop_tolerance', type=float, default=0.5,
                        help='Channel drop tolerance')
    parser.add_argument('--min_width_channels', type=int, default=16,
                        help='Minimum width of channels')
    parser.add_argument('--max_width_channels', type=int, default=64,
                        help='Maximum width of channels')
    parser.add_argument('--width_resolution', type=int, default=4,
                        help='Width resolution')
    parser.add_argument('--min_depth_layers', type=int, default=5,
                        help='Minimum depth of layers')
    parser.add_argument('--max_depth_layers', type=int, default=20,
                        help='Maximum depth of layers')
    args = parser.parse_args()
    return args


def setup_start(config):
    exp_name = config['dataset_to_run']
    txt_name = exp_name + '.txt'
    save_name = 'Search-{}-{}'.format(exp_name, strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(save_name, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_name, txt_name))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return save_name

def main(save_name, config):
    if not torch.cuda.is_available():
        logging.info('GPU not available.')
        sys.exit(1)
    #config = utils.load_yaml()
    device = config['device']['gpu']
    seed = config['hyperparameters']['seed']
    
    grayscale = utils.get_grayscale(config['dataset_to_run'])
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
    # Fetch the desired sub dataset.
    # Run search on the fetched sub dataset.
    results1 = search_model.search_depth_and_width(class_labels,
                                                   trainer.train_test,
                                                   trainer.train)
    curr_arch_ops, curr_arch_kernel, f_channels = results1[:3]
    f_layers, curr_arch_train_acc, curr_arch_test_acc = results1[3:]

    d_w_model_info = {'curr_arch_ops': curr_arch_ops,
                      'curr_arch_kernel': curr_arch_kernel,
                      'curr_arch_train_acc': curr_arch_train_acc,
                      'curr_arch_test_acc': curr_arch_test_acc,
                      'f_channels': f_channels,
                      'f_layers': f_layers}
    results2 = search_model.search_operations_and_kernels(d_w_model_info,
                                                          class_labels,
                                                          trainer.train_test,
                                                          trainer.train)
    curr_arch_ops, curr_arch_kernel = results2[:2]
    curr_arch_train_acc, curr_arch_test_acc = results2[2:]

    logging.info('END OF SEARCH...')
    utils.log_hash()
    input_shape = utils.get_input_shape(config['dataset_to_run'])
    model = NetworkMixArch(f_channels, len(np.array(class_labels)), f_layers,
                           curr_arch_ops, curr_arch_kernel, input_shape,
                           grayscale)
    model = model.cuda()
    logging.info('FINAL DISCOVERED ARCHITECTURE DETAILS:')
    logging.info("Model Depth %s Model Width %s", f_layers, f_channels)
    logging.info("Model Layers %s Model Kernels %s", curr_arch_ops,
                 curr_arch_kernel)
    logging.info("Model Parameters = %fMB",
                 utils.count_parameters_in_MB(model))
    logging.info("Training Accuracy %f Validation Accuracy %f",
                 curr_arch_train_acc, curr_arch_test_acc)
    logging.info('Final model summary: %s', model)


if __name__ == '__main__':
    start_time = time()
    config = utils.load_yaml()
    args = get_args()
    updated_config = update_config_with_args(config, args)
    save_name = setup_start(updated_config)
    main(save_name, updated_config)
    end_time = time()
    duration = end_time - start_time
    logging.info('Total Search Time: %ds', duration)
