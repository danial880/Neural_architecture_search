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

from torch.autograd import Variable
from NetworkMix import NetworkMix_ImageNet
import torchvision.transforms as transforms
from Subsets import *

parser = argparse.ArgumentParser("Neural Architecture search")

parser.add_argument('--dataset', type=str, default='cifar10', help='dataset name')

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

  # Log sub dataset under consideration.

  train_queue, valid_queue, test_queue ,classes, class_labels = get_desired_dataset(args)
  
  print('Classes under consideration: ', classes)
  logging.info("Classes under consideration: %s", classes)
  ImageNet_CLASSES = len(np.array(class_labels))
  # Fetch the desired sub dataset.
  # Run search on the fetched sub dataset.
  curr_arch_ops, curr_arch_kernel, f_channels, f_layers, curr_arch_train_acc, curr_arch_test_acc = search_depth_and_width(args,   
  	                                                                                                                      classes,        
                                                                                                                          train_queue,
                                                                                                                          valid_queue,                                                                                                                          
                                                                                                                          test_queue,
                                                                                                                          class_labels)
  
  d_w_model_info = {'curr_arch_ops': curr_arch_ops,
                    'curr_arch_kernel': curr_arch_kernel,
                    'curr_arch_train_acc': curr_arch_train_acc,
                    'curr_arch_test_acc': curr_arch_test_acc,
                    'f_channels': f_channels,
                    'f_layers': f_layers}

  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_operations_and_kernels(args, 
                                                                                                    classes,
                                                                                                    train_queue,
                                                                                                    valid_queue,
                                                                                                    test_queue, 
                                                                                                    d_w_model_info,
                                                                                                    class_labels)


  logging.info('END OF SEARCH...')
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('')

  model = NetworkMix_ImageNet(f_channels, ImageNet_CLASSES, f_layers, curr_arch_ops, curr_arch_kernel)
  model = model.cuda()                                                                                                                           

  logging.info('FINAL DISCOVERED ARCHITECTURE DETAILS:')
  logging.info("Model Depth %s Model Width %s", f_layers, f_channels)
  logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
  logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
  logging.info("Training Accuracy %f Validation Accuracy %f", curr_arch_train_acc, curr_arch_test_acc)

def search_depth_and_width(args, classes, train_queue, valid_queue, test_queue,class_labels):


  logging.info('INITIALIZING DEPTH AND WIDTH SEARCH...')

  ImageNet_CLASSES = len(np.array(class_labels)) 
  target_acc=args.target_acc
  min_width=args.min_width
  max_width=args.max_width
  width_resolution=args.width_resolution
  min_depth=args.min_depth
  max_depth=args.max_depth
  ch_drop_tolerance = args.ch_drop_tolerance
  target_acc_tolerance = args.target_acc_tolerance
  # We start with max width but with min depth.
  channels = max_width 
  layers = min_depth

  # Initialize
  curr_arch_ops = next_arch_ops = np.zeros((layers,), dtype=int)
  curr_arch_kernel = next_arch_kernel = 3*np.ones((layers,), dtype=int)
  curr_arch_train_acc = next_arch_train_acc = 0.0
  curr_arch_test_acc = next_arch_test_acc = 0.0

  logging.info('RUNNING DEPTH SEARCH FIRST...')

  model = NetworkMix_ImageNet(channels, ImageNet_CLASSES, layers, curr_arch_ops, curr_arch_kernel)
  model = model.cuda()
  
  logging.info('MODEL DETAILS')
  logging.info("Model Depth %s Model Width %s", layers, channels)
  logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
  logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
  logging.info('Training Model...')
  curr_arch_train_acc, curr_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
  logging.info("Best Training Accuracy %f Best Validation Accuracy %f", curr_arch_train_acc, curr_arch_test_acc)

  # Search depth

  while ((curr_arch_test_acc < (target_acc - target_acc_tolerance)) and (layers != max_depth)):
    
    # The possibility exists if trained for too long.
    if (curr_arch_train_acc == 100):
      break;  
      
    else:
      # prepare next candidate architecture.  
      layers += 1
      next_arch_ops = np.zeros((layers,), dtype=int)
      next_arch_kernel = 3*np.ones((layers,), dtype=int)
      model = NetworkMix_ImageNet(channels, ImageNet_CLASSES, layers, next_arch_ops, next_arch_kernel)
      model = model.cuda()
      logging.info('#############################################################################')
      logging.info('Moving to Next Candidate Architecture...')
      logging.info('MODEL DETAILS')
      logging.info("Model Depth %s Model Width %s", layers, channels)
      logging.info("Model Layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
      logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
      logging.info('Training Model...')
      next_arch_train_acc, next_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
      logging.info("Best Training Accuracy %f Best Validation Accuracy %f", next_arch_train_acc, next_arch_test_acc)
     
      # As long as we get significant improvement by increasing depth.
      
      if (next_arch_test_acc >= curr_arch_test_acc + 0.25):
        # update current architecture.
        curr_arch_ops = next_arch_ops
        curr_arch_kernel = next_arch_kernel
        curr_arch_train_acc = next_arch_train_acc
        curr_arch_test_acc = next_arch_test_acc
      # But we still keep trying deeper candidates.
      elif (next_arch_test_acc >= curr_arch_test_acc - 0.15):
        continue
      else:
        break
  # Search width
  # During width search lenght of curr_arch_ops and curr_arch_kernel shall not change but only channels.

  f_layers = len(curr_arch_ops) # discovered final number of layers
  f_channels = max_width # discovered final number of channels
  logging.info('Discovered Final Depth %s', f_layers)
  logging.info('END OF DEPTH SEARCH...')
  best_arch_test_acc = curr_arch_test_acc
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('')
  logging.info('RUNNING WIDTH SEARCH NOW...') 
  while (channels > min_width):
    # prepare next candidate architecture.
    channels = channels - width_resolution
    # Although these do not change.
    model = NetworkMix_ImageNet(channels, ImageNet_CLASSES, f_layers, curr_arch_ops, curr_arch_kernel)
    model = model.cuda()
    logging.info('Moving to Next Candidate Architecture...')
    logging.info('MODEL DETAILS')
    logging.info("Model Depth %s Model Width %s", layers, channels)
    logging.info("Model Layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
    logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
    logging.info('Training Model...')
    # train and test candidate architecture.
    next_arch_train_acc, next_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
    logging.info("Best Training Accuracy %f Best Validation Accuracy %f", next_arch_train_acc, next_arch_test_acc)

    if (next_arch_test_acc >= (best_arch_test_acc - ch_drop_tolerance)):
      curr_arch_train_acc = next_arch_train_acc
      curr_arch_test_acc = next_arch_test_acc
      f_channels = channels 
    else:
      break; 

  logging.info('Discovered Final Width %s', f_channels)
  logging.info('END OF WIDTH SEARCH...')  
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('')  

  return curr_arch_ops, curr_arch_kernel, f_channels, f_layers, curr_arch_train_acc, curr_arch_test_acc

def search_operations(args, classes, train_queue, valid_queue, test_queue, model_info,class_labels):

  logging.info('RUNNING OPERATION SEARCH...')

  ImageNet_CLASSES = len(np.array(class_labels)) 
  curr_arch_ops = model_info['curr_arch_ops']
  curr_arch_kernel = model_info['curr_arch_kernel']
  curr_arch_train_acc = model_info['curr_arch_train_acc']
  curr_arch_test_acc = model_info['curr_arch_test_acc']
  channels = model_info['f_channels']
  layers = model_info['f_layers']

  next_arch_ops = curr_arch_ops
  next_arch_kernel = curr_arch_kernel    

  for i in range(layers):
 
    next_arch_ops[i] = 1

    model = NetworkMix_ImageNet(channels, ImageNet_CLASSES, layers, next_arch_ops, next_arch_kernel)
    model = model.cuda()
  
    logging.info('NEXT MODEL DETAILS')
    logging.info("Model Depth %s Model Width %s", layers, channels)
    logging.info("Model Layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
    logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
    logging.info('Training Model...')
    next_arch_train_acc, next_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
    logging.info("Best Training Accuracy %f Best Validation Accuracy %f", next_arch_train_acc, next_arch_test_acc)

    if next_arch_test_acc > curr_arch_test_acc:
      curr_arch_ops = next_arch_ops
      curr_arch_kernel = next_arch_kernel
      curr_arch_train_acc = next_arch_train_acc
      curr_arch_test_acc = next_arch_test_acc
    else:
      next_arch_ops[i] = 0

  logging.info('Discovered Final Operations %s', curr_arch_ops)
  logging.info('END OF OPERATION SEARCH...')  
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('')  


  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc  

def search_kernels(args, classes, train_queue, valid_queue, test_queue, model_info,class_labels):

  logging.info('RUNNING KERNEL SEARCH...')

  ImageNet_CLASSES = len(np.array(class_labels)) 
  curr_arch_ops = model_info['curr_arch_ops']
  curr_arch_kernel = model_info['curr_arch_kernel']
  curr_arch_train_acc = model_info['curr_arch_train_acc']
  curr_arch_test_acc = model_info['curr_arch_test_acc']
  channels = model_info['f_channels']
  layers = model_info['f_layers']

  next_arch_ops = curr_arch_ops
  next_arch_kernel = curr_arch_kernel

  kernels = [5, 7]
  

  for i in range(layers): 
    best_k = 3 
    for k in kernels:

      next_arch_kernel[i] = k
 
      model = NetworkMix_ImageNet(channels, ImageNet_CLASSES, layers, next_arch_ops, next_arch_kernel)
      model = model.cuda()
  
      logging.info('MODEL DETAILS')
      logging.info("Model Depth %s Model Width %s", layers, channels)
      logging.info("Model Layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
      logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
      logging.info('Training Model...')
      next_arch_train_acc, next_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
      logging.info("Best Training Accuracy %f Best Validation Accuracy %f", next_arch_train_acc, next_arch_test_acc)


      # Bigger kernel comes at a cost therefore possibility of a search hyper parameter exists.
      if (next_arch_test_acc > curr_arch_test_acc):
        best_k = k
        curr_arch_ops = next_arch_ops
        curr_arch_kernel[i] = k
        curr_arch_train_acc = next_arch_train_acc
        curr_arch_test_acc = next_arch_test_acc
      else:
        next_arch_kernel[i] = best_k

  logging.info('Discovered Final Kernels %s', curr_arch_kernel)
  logging.info('END OF KERNEL SEARCH...')  
  logging.info('#############################################################################')
  logging.info('#############################################################################')
  logging.info('')          
  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc

def search_ops_and_ks_simultaneous(args, classes, train_queue, valid_queue, test_queue, model_info,class_labels):

  logging.info('RUNNING OPERATIONS AND KERNELS SEARCH SIMULTANEOUSLY...')

  ImageNet_CLASSES = len(np.array(class_labels)) 
  curr_arch_ops = model_info['curr_arch_ops']
  curr_arch_kernel = model_info['curr_arch_kernel']
  curr_arch_train_acc = model_info['curr_arch_train_acc']
  curr_arch_test_acc = model_info['curr_arch_test_acc']
  channels = model_info['f_channels']
  layers = model_info['f_layers']

  kernels = [3, 5, 7]
  operations = [0, 1]

  next_arch_ops = curr_arch_ops
  next_arch_kernel = curr_arch_kernel
  # Can be navigated from the last layers instead of first ones.
  for i in range(layers):  
    for k in kernels:
      for o in operations:


        next_arch_ops[i] = o
        next_arch_kernel[i] = k
 
        model = NetworkMix_ImageNet(channels, ImageNet_CLASSES, layers, next_arch_ops, next_arch_kernel)
        model = model.cuda()
  
        logging.info('MODEL DETAILS')
        logging.info("Model Depth %s Model Width %s", layers, channels)
        logging.info("Model Layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
        logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
        logging.info('Training Model...')
        next_arch_train_acc, next_arch_test_acc = train_test(args, classes, model, train_queue, valid_queue, test_queue)
        logging.info("Best Training Accuracy %f Best Validation Accuracy %f", next_arch_train_acc, next_arch_test_acc)

        # Bigger kernel comes at a cost therefore possibility of a search hyper parameter exists.
        if (next_arch_test_acc > curr_arch_test_acc):
          curr_arch_ops = next_arch_ops
          curr_arch_kernel = next_arch_kernel
          curr_arch_train_acc = next_arch_train_acc
          curr_arch_test_acc = next_arch_test_acc

  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc    

def search_kernels_and_operations(args, classes, train_queue, valid_queue, test_queue, model_info,class_labels):

  logging.info('SEARCHING FOR KERNELS FIRST...')

  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_kernels(args, classes, train_queue, valid_queue, test_queue, model_info,class_labels)
  
  model_info['curr_arch_ops'] = curr_arch_ops
  model_info['curr_arch_kernel'] = curr_arch_kernel
  model_info['curr_arch_train_acc'] = curr_arch_train_acc
  model_info['curr_arch_test_acc'] = curr_arch_test_acc

  logging.info('SEARCHING FOR OPERATIONS...')

  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_operations(args, classes, train_queue, valid_queue, test_queue, model_info)

  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc

def search_operations_and_kernels(args, classes, train_queue, valid_queue, test_queue, model_info,class_labels):
  
  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_operations(args, classes, train_queue, valid_queue, test_queue, model_info,class_labels)
  
  model_info['curr_arch_ops'] = curr_arch_ops
  model_info['curr_arch_kernel'] = curr_arch_kernel
  model_info['curr_arch_train_acc'] = curr_arch_train_acc
  model_info['curr_arch_test_acc'] = curr_arch_test_acc
  
  curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_kernels(args, classes, train_queue, valid_queue, test_queue, model_info,class_labels)

  return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc

def get_desired_dataset(args):

  # traindir = os.path.join(args.data, 'train')
  # validdir = os.path.join(args.data, 'val')

  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  train_transform= transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
              brightness=0.4,
              contrast=0.4,
              saturation=0.4,
              hue=0.2),
            transforms.ToTensor(),
            normalize,
          ])

  test_transform = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ])
  
  if args.dataset == 'cifar10':
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
  
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)
    
    total_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
   
  if args.dataset == 'STL10':
    train_data = torchvision.datasets.STL10(root='./data', train=True,
                                        download=True, transform=train_transform)
   
    test_data = torchvision.datasets.STL10(root='./data', train=False,
                                           download=True, transform=test_transform)
   
    total_classes = ['airplane','bird','car','cat','deer','dogs','horse','monkey','ship','truck']
   
  
  if args.dataset == 'DTD':

    train_data = torchvision.datasets.DTD(root='./data', split='train',download=True, transform=train_transform)
   
    test_data = torchvision.datasets.DTD(root='./data', split='val',
                                           download=True, transform=test_transform)
    total_classes = ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered', 'cobwebbed', 'cracked', 'crosshatched',  
              'crystalline', 'dotted', 'fibrous', 'flecked', 'frothy', 'gauzy', 'grid', 'grooved', 'herringbone',   
               'interlaced', 'knitted', 'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley', 'patterned',  
               'plaid', 'polka-dotted', 'porous', 'radial', 'ribbed', 'scaled', 'smeared', 'spangled', 'speckled',
               'stippled', 'striped', 'studded', 'swirly', 'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged',
                'homogeneous', 'non-homogeneous']
    
  if args.dataset == 'cifar100':
    
    train_data = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=train_transform)
   
    test_data = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=test_transform)
   
    total_classes =['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',            
    'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',            
    'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',            
    'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',            
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',            
    'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',            
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',            
    'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',            
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',            
    'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    
    
  if args.dataset == 'Food101':
    train_data = torchvision.datasets.Food101(root='./data', split='train',download=True, transform=train_transform)
   
    test_data = torchvision.datasets.Food101(root='./data', split='val',download=True, transform=test_transform)

    total_classes = ['macarons', 'french_toast', 'lobster_bisque', 'prime_rib', 'pork_chop', 'guacamole', 'baby_back_ribs', 'mussels', 'beef_carpaccio', 'poutine',
                    'hot_and_sour_soup', 'seaweed_salad', 'foie_gras', 'dumplings', 'peking_duck', 'takoyaki', 'bibimbap', 'falafel', 'pulled_pork_sandwich', 'lobster_roll_sandwich',
                    'carrot_cake', 'beet_salad', 'panna_cotta', 'donuts', 'red_velvet_cake', 'grilled_cheese_sandwich', 'cannoli', 'spring_rolls', 'shrimp_and_grits',
                    'clam_chowder','omelette', 'fried_calamari', 'caprese_salad', 'oysters', 'scallops', 'ramen', 'grilled_salmon', 'croque_madame', 'filet_mignon',
                    'hamburger', 'spaghetti_carbonara', 'miso_soup', 'bread_pudding', 'lasagna', 'crab_cakes', 'cheesecake', 'spaghetti_bolognese', 'cup_cakes', 'creme_brulee',
                    'waffles', 'fish_and_chips', 'paella', 'macaroni_and_cheese', 'chocolate_mousse', 'ravioli', 'chicken_curry', 'caesar_salad', 'nachos', 'tiramisu', 'frozen_yogurt',
                    'ice_cream', 'risotto', 'club_sandwich', 'strawberry_shortcake', 'steak', 'churros', 'garlic_bread', 'baklava', 'bruschetta', 'hummus', 'chicken_wings',
                    'greek_salad', 'tuna_tartare', 'chocolate_cake', 'gyoza', 'eggs_benedict', 'deviled_eggs', 'samosa', 'sushi', 'breakfast_burrito', 'ceviche', 'beef_tartare',
                    'apple_pie', '.DS_Store', 'huevos_rancheros', 'beignets', 'pizza', 'edamame', 'french_onion_soup', 'hot_dog', 'tacos', 'chicken_quesadilla', 'pho', 'gnocchi',
                    'pancakes', 'fried_rice', 'cheese_plate', 'onion_rings', 'escargots', 'sashimi', 'pad_thai', 'french_fries']

  if args.dataset == 'flower102':

    train_data = torchvision.datasets.Flowers102(root='./data', split='train',
                                        download=True, transform=train_transform)
   
    test_data = torchvision.datasets.Flowers102(root='./data', split='val',
                                           download=True, transform=test_transform)

    total_classes =["pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold", 
    "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon",
     "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
    "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox", 
    "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort", 
    "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue", "wallflower", 
    "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower", "pelargonium", 
    "bishop of llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone", 
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", 
    "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus", "toad lily", "anthurium", 
    "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress", 
    "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", 
    "bromelia", "blanket flower", "trumpet creeper", "blackberry lily"]
  
  

  # obtain training indices that will be used for validation
  valid_size = args.valid_size
  num_train = len(train_data)
  indices = list(range(num_train))
  np.random.shuffle(indices)
  split = int(np.floor(valid_size * num_train))
  train_idx, valid_idx = indices[split:], indices[:split]

  # define samplers for obtaining training and validation batches
  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)

  train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,sampler=train_sampler, num_workers=2)
  valid_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,sampler=valid_sampler, num_workers=2)

  test_queue = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  # get some random training images
  dataiter = iter(train_queue)
  images, labels = next(dataiter)
  print("***** Shape of the dataset *******",images.size())
  classes = []

  class_labels = []
  for i in range(len(total_classes)):
    class_labels.append(i)

  for i in np.array(class_labels):
    classes.append(total_classes[i])

  return train_queue, valid_queue, test_queue, classes, class_labels

def train_test(args, classes, model, train_queue, valid_queue, test_queue):
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  best_train_acc = 0.0
  best_test_acc = 0.0

  for epoch in range(args.epochs):
    scheduler.step()    

    start_time = time.time()
    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    #logging.info('train_acc %f', train_acc)

    if args.valid_size == 0:
      valid_acc, valid_obj = infer(test_queue, model, criterion)
    else:
      valid_acc, valid_obj = infer(valid_queue, model, criterion)

    if epoch % args.report_freq == 0:
      logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])	
      logging.info('train_acc %f', train_acc)  
      logging.info('valid_acc %f', valid_acc)
       

    end_time = time.time()
    duration = end_time - start_time
    print('Epoch time: %ds.' %duration)
    print('Train acc: %f ' %train_acc)
    print('Valid_acc: %f ' %valid_acc)

    if train_acc > best_train_acc:
      best_train_acc = train_acc

    if valid_acc > best_test_acc:
      best_test_acc = valid_acc
      utils.save(model, os.path.join(args.save, 'weights.pt'))

    #if best_train_acc == 100:
    #	break
    	

  logging.info('Best Training Accuracy %f', best_train_acc)
  logging.info('Best Validation Accuracy %f', best_test_acc)
  utils.load(model, os.path.join(args.save, 'weights.pt'))
  classwisetest(model, classes, test_queue, criterion)

  return best_train_acc, best_test_acc

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()

  model.train()

  for step, (input, target) in enumerate(train_queue):

    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1 = utils.accuracy(logits, target, topk=(1,))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    #top1.update(prec1.data.item(), n)
    top1.update(prec1, n)

    #if step % args.report_freq == 0:
    #  logging.info('train %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()

  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    with torch.no_grad():
      logits = model(input)
      loss = criterion(logits, target)

    prec1 = utils.accuracy(logits, target, topk=(1,))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    #top1.update(prec1.data.item(), n)
    top1.update(prec1, n)
    #if step % args.report_freq == 0:
    #  logging.info('valid %03d %e %f', step, objs.avg, top1.avg)

  return top1.avg, objs.avg

def classwisetest(model, classes, test_queue, criterion):

    
    num_classes = len(classes)
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    model.eval()
    # iterate over test data
    for data, target in test_queue:
        # move tensors to GPU if CUDA is available        
        data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)    
        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        #correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        
        # calculate test accuracy for each object class
        for i in range(len(target)):
            #print(i)
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # average test loss
    test_loss = test_loss/len(test_queue.dataset)
    
    logging.info('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(num_classes):
        if class_total[i] > 0:
            logging.info('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            logging.info('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    logging.info('\nTest Accuracy (Overall): %2f%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))    


if __name__ == '__main__':
  start_time = time.time()
  main() 
  end_time = time.time()
  duration = end_time - start_time
  logging.info('Total Search Time: %ds', duration)