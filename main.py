import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn
import torchvision
from torchsummary import summary

from torch.autograd import Variable
from NetworkMix import NetworkMix_ImageNet, NetworkMix_arch
import torchvision.transforms as transforms
from Subsets import *
from trainer import Train
from model_search import Model

parser = argparse.ArgumentParser("Neural Architecture search")

parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')
parser.add_argument('--data_dir', type=str, default='./data', help='dataset path')
parser.add_argument('--input_shape', '--list', type=int, nargs='*',default=[224,224], help='dataset shape')

parser.add_argument('--valid_size', type=float, default=0, help='validation data size')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs') 

parser.add_argument('--report_freq', type=float, default=30, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='hard-ds', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')#0

parser.add_argument('--target_acc', type=float, default=99.00, help='desired target accuracy')
parser.add_argument('--target_acc_tolerance', type=float, default=0.5, help='tolerance for desired target accuracy')
parser.add_argument('--ch_drop_tolerance', type=float, default=0.5, help='tolerance when dropping channels')
parser.add_argument('--min_width', type=int, default=16, help='minimum number of init channels in search')
parser.add_argument('--max_width', type=int, default=64, help='maximum number of init channels in search')
parser.add_argument('--width_resolution', type=int, default=16, help='resolution for number of channels search')
parser.add_argument('--min_depth', type=int, default=5, help='minimum number of init layers in search')
parser.add_argument('--max_depth', type=int, default=15, help='maximum number of init layers in search')

args = parser.parse_args()
def setup_start(args):

  args.save = 'Search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'Searchlog.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)


def main():
  if not torch.cuda.is_available():
    logging.info('GPU not available.')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('GPU Device = %d' % args.gpu)
  logging.info("Arguments = %s", args)
  trainer = Train(args)
  # Log sub dataset under consideration.
  class_labels = trainer.get_desired_dataset(args,logging)
  search_model = Model(args,class_labels)
  
  
  # Fetch the desired sub dataset.
  # Run search on the fetched sub dataset.
  curr_arch_ops, curr_arch_kernel, f_channels, f_layers, curr_arch_train_acc, curr_arch_test_acc = search_model.search_depth_and_width(args,   
  	                                                                                                                      
                                                                                                                          class_labels,trainer.train_test,trainer.train)
  
  d_w_model_info = {'curr_arch_ops': curr_arch_ops,
                    'curr_arch_kernel': curr_arch_kernel,
                    'curr_arch_train_acc': curr_arch_train_acc,
                    'curr_arch_test_acc': curr_arch_test_acc,
                    'f_channels': f_channels,
                    'f_layers': f_layers}

  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_model.search_operations_and_kernels(args, d_w_model_info, 
                                                                                                     class_labels,trainer.train_test,trainer.train)


  logging.info('END OF SEARCH...')
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('')

  # model = NetworkMix_ImageNet(f_channels, ImageNet_CLASSES, f_layers, curr_arch_ops, curr_arch_kernel)
  model = NetworkMix_arch(f_channels,  len(np.array(class_labels)) , f_layers, curr_arch_ops, curr_arch_kernel,args.input_shape)

  model = model.cuda()                                                                                                                           

  logging.info('FINAL DISCOVERED ARCHITECTURE DETAILS:')
  logging.info("Model Depth %s Model Width %s", f_layers, f_channels)
  logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
  logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
  logging.info("Training Accuracy %f Validation Accuracy %f", curr_arch_train_acc, curr_arch_test_acc)

if __name__ == '__main__':
  start_time = time.time()
  setup_start(args)
  main() 
  end_time = time.time()
  duration = end_time - start_time
  logging.info('Total Search Time: %ds', duration)