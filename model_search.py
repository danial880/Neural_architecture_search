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
from NetworkMix import  NetworkMix_arch
import torchvision.transforms as transforms
from Subsets import *

class Model():
    def __init__(self, args, class_labels):
        self.CLASSES = len(np.array(class_labels)) 
        self.target_acc=args.target_acc
        self.min_width=args.min_width
        self.max_width=args.max_width
        self.width_resolution=args.width_resolution
        self.min_depth=args.min_depth
        self.max_depth=args.max_depth
        self.ch_drop_tolerance = args.ch_drop_tolerance
        self.target_acc_tolerance = args.target_acc_tolerance
        # We start with max width but with min depth.
        self.channels = self.max_width 
        self.layers = self.min_depth
        self.input_shape = args.input_shape

    def search_depth_and_width(self,args,class_labels,train_test,train):

        logging.info('INITIALIZING DEPTH AND WIDTH SEARCH...')

        # Initialize
        curr_arch_ops = next_arch_ops = np.zeros((self.layers,), dtype=int)
        curr_arch_kernel = next_arch_kernel = 3*np.ones((self.layers,), dtype=int)
        curr_arch_train_acc = next_arch_train_acc = 0.0
        curr_arch_test_acc = next_arch_test_acc = 0.0

        logging.info('RUNNING DEPTH SEARCH FIRST...')

        # model = NetworkMix_ImageNet(channels, self.CLASSES, self.layers, curr_arch_ops, curr_arch_kernel)
        model = NetworkMix_arch(self.channels, self.CLASSES, self.layers, curr_arch_ops, curr_arch_kernel,self.input_shape)
        print(model)
        model = model.cuda()

        logging.info('MODEL DETAILS')
        logging.info("Model Depth %s Model Width %s", self.layers, self.channels)
        logging.info("Model self.layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
        logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
        logging.info('Training Model...')
        curr_arch_train_acc, curr_arch_test_acc = train_test(args, model)
        logging.info("Best Training Accuracy %f Best Validation Accuracy %f", curr_arch_train_acc, curr_arch_test_acc)

        # Search depth

        while ((curr_arch_test_acc < (self.target_acc - self.target_acc_tolerance)) and (self.layers != self.max_depth)):

            # The possibility exists if trained for too long.
            if (curr_arch_train_acc == 100):
                break;  
                
            else:
                # prepare next candidate architecture.  
                self.layers += 1
                next_arch_ops = np.zeros((self.layers,), dtype=int)
                next_arch_kernel = 3*np.ones((self.layers,), dtype=int)
                # model = NetworkMix_ImageNet(channels, self.CLASSES, self.layers, next_arch_ops, next_arch_kernel)
                model = NetworkMix_arch(self.channels, self.CLASSES, self.layers, next_arch_ops, next_arch_kernel,self.input_shape) 
                model = model.cuda()
                logging.info('#############################################################################')
                logging.info('Moving to Next Candidate Architecture...')
                logging.info('MODEL DETAILS')
                logging.info("Model Depth %s Model Width %s", self.layers, self.channels)
                logging.info("Model self.layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
                logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
                logging.info('Training Model...')
                next_arch_train_acc, next_arch_test_acc = train_test(args, model)
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

        f_layers = len(curr_arch_ops) # discovered final number of self.layers
        f_channels = self.max_width # discovered final number of channels
        logging.info('Discovered Final Depth %s', f_layers)
        logging.info('END OF DEPTH SEARCH...')
        best_arch_test_acc = curr_arch_test_acc
        logging.info('#############################################################################')
        logging.info('#############################################################################')
        logging.info('')
        logging.info('RUNNING WIDTH SEARCH NOW...') 
        while (self.channels > self.min_width):
            # prepare next candidate architecture.
            self.channels = self.channels - self.width_resolution
            # Although these do not change.
            # model = NetworkMix_ImageNet(channels, self.CLASSES, f_self.layers, curr_arch_ops, curr_arch_kernel)
            model = NetworkMix_arch(self.channels, self.CLASSES, f_layers, curr_arch_ops, curr_arch_kernel,self.input_shape)
            model = model.cuda()
            logging.info('Moving to Next Candidate Architecture...')
            logging.info('MODEL DETAILS')
            logging.info("Model Depth %s Model Width %s", self.layers, self.channels)
            logging.info("Model self.layers %s Model Kernels %s", curr_arch_ops, curr_arch_kernel)
            logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
            logging.info('Training Model...')
            # train and test candidate architecture.
            next_arch_train_acc, next_arch_test_acc = train_test(args,  model)
            logging.info("Best Training Accuracy %f Best Validation Accuracy %f", next_arch_train_acc, next_arch_test_acc)

            if (next_arch_test_acc >= (best_arch_test_acc - self.ch_drop_tolerance)):
                curr_arch_train_acc = next_arch_train_acc
                curr_arch_test_acc = next_arch_test_acc
                f_channels = self.channels 
            else:
                break; 

        logging.info('Discovered Final Width %s', f_channels)
        logging.info('END OF WIDTH SEARCH...')  
        logging.info('#############################################################################')
        logging.info('#############################################################################')
        logging.info('')  

        return curr_arch_ops, curr_arch_kernel, f_channels, f_layers, curr_arch_train_acc, curr_arch_test_acc

    def search_operations(self,args,class_labels,train_test,train ,model_info):

        logging.info('RUNNING OPERATION SEARCH...')

        curr_arch_ops = model_info['curr_arch_ops']
        curr_arch_kernel = model_info['curr_arch_kernel']
        curr_arch_train_acc = model_info['curr_arch_train_acc']
        curr_arch_test_acc = model_info['curr_arch_test_acc']
        channels = model_info['f_channels']
        self.layers = model_info['f_layers']

        next_arch_ops = curr_arch_ops
        next_arch_kernel = curr_arch_kernel    

        for i in range(self.layers):

            next_arch_ops[i] = 1

            # model = NetworkMix_ImageNet(channels, self.CLASSES, self.layers, next_arch_ops, next_arch_kernel)
            model = NetworkMix_arch(channels, self.CLASSES, self.layers, next_arch_ops, next_arch_kernel,self.input_shape)
            model = model.cuda()

            logging.info('NEXT MODEL DETAILS')
            logging.info("Model Depth %s Model Width %s", self.layers, channels)
            logging.info("Model self.layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
            logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
            logging.info('Training Model...')
            next_arch_train_acc, next_arch_test_acc = train_test(args, model)
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

    def search_kernels(self,args,class_labels,train_test,train ,model_info):

        logging.info('RUNNING KERNEL SEARCH...')

        curr_arch_ops = model_info['curr_arch_ops']
        curr_arch_kernel = model_info['curr_arch_kernel']
        curr_arch_train_acc = model_info['curr_arch_train_acc']
        curr_arch_test_acc = model_info['curr_arch_test_acc']
        self.channels = model_info['f_channels']
        self.layers = model_info['f_layers']

        next_arch_ops = curr_arch_ops
        next_arch_kernel = curr_arch_kernel

        kernels = [5, 7]


        for i in range(self.layers): 
            best_k = 3 
            for k in kernels:

                next_arch_kernel[i] = k

                # model = NetworkMix_ImageNet(channels, self.CLASSES, self.layers, next_arch_ops, next_arch_kernel)
                model =  NetworkMix_arch(self.channels, self.CLASSES, self.layers, next_arch_ops, next_arch_kernel,self.input_shape)
                model = model.cuda()

                logging.info('MODEL DETAILS')
                logging.info("Model Depth %s Model Width %s", self.layers, self.channels)
                logging.info("Model self.layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
                logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
                logging.info('Training Model...')
                next_arch_train_acc, next_arch_test_acc = train_test(args, model)
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

    # def search_ops_and_ks_simultaneous(self, args,class_labels,train_test,train ,model_info):

    #     logging.info('RUNNING OPERATIONS AND KERNELS SEARCH SIMULTANEOUSLY...')

    #     curr_arch_ops = model_info['curr_arch_ops']
    #     curr_arch_kernel = model_info['curr_arch_kernel']
    #     curr_arch_train_acc = model_info['curr_arch_train_acc']
    #     curr_arch_test_acc = model_info['curr_arch_test_acc']
    #     self.channels = model_info['f_channels']
    #     self.layers = model_info['f_self.layers']

    #     kernels = [3, 5, 7]
    #     operations = [0, 1]

    #     next_arch_ops = curr_arch_ops
    #     next_arch_kernel = curr_arch_kernel
    #     # Can be navigated from the last self.layers instead of first ones.
    #     for i in range(self.layers):  
    #         for k in kernels:
    #             for o in operations:


    #                 next_arch_ops[i] = o
    #                 next_arch_kernel[i] = k

    #                 # model = NetworkMix_ImageNet(channels, self.CLASSES, self.layers, next_arch_ops, next_arch_kernel)
    #                 model = NetworkMix_arch(self.channels, self.CLASSES, self.layers, next_arch_ops, next_arch_kernel,self.input_shape)

    #                 model = model.cuda()

    #                 logging.info('MODEL DETAILS')
    #                 logging.info("Model Depth %s Model Width %s", self.layers, self.channels)
    #                 logging.info("Model self.layers %s Model Kernels %s", next_arch_ops, next_arch_kernel)
    #                 logging.info("Model Parameters = %fMB", utils.count_parameters_in_MB(model))
    #                 logging.info('Training Model...')
    #                 next_arch_train_acc, next_arch_test_acc = train_test(args,  model,)
    #                 logging.info("Best Training Accuracy %f Best Validation Accuracy %f", next_arch_train_acc, next_arch_test_acc)

    #                 # Bigger kernel comes at a cost therefore possibility of a search hyper parameter exists.
    #                 if (next_arch_test_acc > curr_arch_test_acc):
    #                     curr_arch_ops = next_arch_ops
    #                     curr_arch_kernel = next_arch_kernel
    #                     curr_arch_train_acc = next_arch_train_acc
    #                     curr_arch_test_acc = next_arch_test_acc

    #     return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc    

    # def search_kernels_and_operations(self, args,class_labels,train_test,train ,model_info):

    #     logging.info('SEARCHING FOR KERNELS FIRST...')

    #     curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_kernels(args,class_labels,train_test,train ,model_info)

    #     model_info['curr_arch_ops'] = curr_arch_ops
    #     model_info['curr_arch_kernel'] = curr_arch_kernel
    #     model_info['curr_arch_train_acc'] = curr_arch_train_acc
    #     model_info['curr_arch_test_acc'] = curr_arch_test_acc

    #     logging.info('SEARCHING FOR OPERATIONS...')

    #     curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = search_operations(args,class_labels,train_test,train ,model_info)

    #     return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc

    def search_operations_and_kernels(self, args,model_info, class_labels, train_test, train ):

        curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = self.search_operations(args, class_labels,train_test,train ,model_info)

        model_info['curr_arch_ops'] = curr_arch_ops
        model_info['curr_arch_kernel'] = curr_arch_kernel
        model_info['curr_arch_train_acc'] = curr_arch_train_acc
        model_info['curr_arch_test_acc'] = curr_arch_test_acc

        curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc = self.search_kernels(args,class_labels,train_test,train ,model_info)

        return curr_arch_ops, curr_arch_kernel, curr_arch_train_acc, curr_arch_test_acc
