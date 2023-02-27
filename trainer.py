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
from NetworkMix import NetworkMix_arch
import torchvision.transforms as transforms
from Subsets import *

class Train():
  def __init__(self, args):

    self.batch_size = args.batch_size
    self.valid_size = args.valid_size
    self.classes = []

    self.class_labels = []

  def get_desired_dataset(self, args,logging):

    
    train_data, test_data, total_classes = utils.data_load_transforms(args)
    # obtain training indices that will be used for validation
    
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(self.valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    self.train_queue = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,sampler=train_sampler, num_workers=2)
    self.valid_queue = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size,sampler=valid_sampler, num_workers=2)

    self.test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # get some random training images
    dataiter = iter(self.train_queue)
    images, labels = next(dataiter)
    print("***** Shape of the dataset *******",images.size())
    
    for i in range(len(total_classes)):
      self.class_labels.append(i)

    for i in np.array(self.class_labels):
      self.classes.append(total_classes[i])

    print('Classes under consideration: ', self.classes)
    logging.info("Classes under consideration: %s", self.classes)

    # return train_queue, valid_queue, test_queue, classes, class_labels
    return self.class_labels


  def train_test(self, args,  model):
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    best_train_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(args.epochs):
      scheduler.step()    

      start_time = time.time()
      train_acc, train_obj = self.train(args,  model, criterion, optimizer)
      #logging.info('train_acc %f', train_acc)

      if self.valid_size == 0:
        valid_acc, valid_obj = self.infer(args, self.test_queue, model, criterion)
      else:
        valid_acc, valid_obj = self.infer(args, self.valid_queue, model, criterion)

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
    self.classwisetest(model, criterion)

    return best_train_acc, best_test_acc

  def train(self, args, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    model.train()

    for step, (input, target) in enumerate(self.train_queue):

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

  def infer(self, args, data,  model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    model.eval()

    for step, (input, target) in enumerate(data):
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

  def classwisetest(self,  model, criterion):

      
      num_classes = len(self.classes)
      # track test loss
      test_loss = 0.0
      class_correct = list(0. for i in range(num_classes))
      class_total = list(0. for i in range(num_classes))
      
      model.eval()
      # iterate over test data
      for data, target in self.test_queue:
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
      test_loss = test_loss/len(self.test_queue.dataset)
      
      logging.info('Test Loss: {:.6f}\n'.format(test_loss))

      for i in range(num_classes):
          if class_total[i] > 0:
              logging.info('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                  self.classes[i], 100 * class_correct[i] / class_total[i],
                  np.sum(class_correct[i]), np.sum(class_total[i])))
          else:
              logging.info('Test Accuracy of %5s: N/A (no training examples)' % (self.classes[i]))

      logging.info('\nTest Accuracy (Overall): %2f%% (%2d/%2d)' % (
          100. * np.sum(class_correct) / np.sum(class_total),
          np.sum(class_correct), np.sum(class_total)))    
