import os
import time
import torch
import utils
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler


class Train():
    """
    Class for training models on a dataset.

    Args:
        args: A namespace object containing the command line arguments.

    Attributes:
        batch_size (int): The batch size used for training.
        valid_size (float): The proportion of data used for validation.
        dataset (str): The name of the dataset being used for training.
        classes (list): A list of classes in the dataset.
        class_labels (list): A list of labels assigned to the classes.
    """
    def __init__(self, config, save_name):
        self.config = config
        self.hyperparameters = self.config['hyperparameters']
        self.save_name = save_name
        self.batch_size = self.hyperparameters['batch_size']
        self.valid_size = self.hyperparameters['valid_size']
        self.dataset = self.config['dataset_to_run']
        self.classes = self.config['datasets'][self.dataset]['classes']
        self.class_labels = list(np.arange(len(self.classes)))

    def get_desired_dataset(self, logging):
        """
        Retrieves the desired dataset for training.

        Args:
            args: A namespace object containing the command line arguments.
            logging: An instance of the logging module.

        Returns:
            A list of class labels for the dataset.
        """
        train_data, test_data, total_classes = utils.data_load_transforms(self.config)
        # obtain training indices that will be used for validation
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        self.train_queue = DataLoader(train_data, batch_size=self.batch_size,
                                      sampler=train_sampler, num_workers=2)
        self.valid_queue = DataLoader(test_data, batch_size=self.batch_size,
                                      sampler=valid_sampler, num_workers=2)
        self.test_queue = DataLoader(test_data, batch_size=self.batch_size,
                                     shuffle=False, pin_memory=True,
                                     num_workers=2)
        # get some random training images
        dataiter = iter(self.train_queue)
        images, labels = next(dataiter)
        print("***** Shape of the dataset *******", images.size())
        #print('Classes under consideration: ', self.classes)
        logging.info("Classes under consideration: %s", self.classes)
        # return train_queue, valid_queue, test_queue, classes, class_labels
        return self.class_labels

    def train_test(self,  model):
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        optimizer = torch.optim.SGD(model.parameters(),
                                    self.hyperparameters['learning_rate'],
                                    momentum=self.hyperparameters['momentum'],
                                    weight_decay=eval(self.hyperparameters['weight_decay']))
        epochs = self.hyperparameters['epochs']
        best_train_acc = 0.0
        if self.config['flags']['resume']:
            print('initializing resume')
            checkpoint = torch.load(self.config['paths']['resume_checkpoint'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_train_acc = checkpoint['hyperparameters']['best_train_acc']
        scheduler = CosineAnnealingLR(optimizer, float(epochs))
        
        best_test_acc = 0.0
        report_freq = self.config['logging']['report_freq']
        start_epoch = checkpoint['epoch'] + 1 if self.config['flags']['resume'] else 0
        for epoch in tqdm(range(start_epoch, epochs)):
            start_time = time.time()
            train_acc, train_obj = self.train(model, criterion, optimizer)           
            # logging.info('train_acc %f', train_acc)
            if self.valid_size == 0:
                valid_acc, valid_obj = self.infer(self.test_queue, model,
                                                  criterion)
            else:
                valid_acc, valid_obj = self.infer(self.valid_queue,
                                                  model, criterion)
            scheduler.step()
            if epoch % report_freq == 0:
                logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
                logging.info('train_acc %f', train_acc)
                logging.info('valid_acc %f', valid_acc)
            end_time = time.time()
            duration = end_time - start_time
            print('Epoch time: %ds.' % duration)
            print('Train acc: %f ' % train_acc)
            print('Valid_acc: %f ' % valid_acc)
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            if (valid_acc > best_test_acc) & self.config['flags']['save']:
                best_test_acc = valid_acc
                utils.save(model, os.path.join(self.save_name, 'weights.pt'))
                checkpoint = {
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'hyperparameters' : {
                    'learning_rate' : self.hyperparameters['learning_rate'],
                    'momentum' : self.hyperparameters['momentum'],
                    'weight_decay' : eval(self.hyperparameters['weight_decay']),
                    'batch_size' : self.hyperparameters['batch_size'],
                    'num_epochs' : epochs,
                    'grad_clip' : self.hyperparameters['grad_clip'],
                    'best_train_acc': best_train_acc
                                }
                         }
                torch.save(checkpoint, 
                           os.path.join(self.save_name, 'checkpoint.pt'))
        logging.info('Best Training Accuracy %f', best_train_acc)
        logging.info('Best Validation Accuracy %f', best_test_acc)
        if self.config['flags']['save']:
            utils.load(model, os.path.join(self.save_name, 'weights.pt'))
            self.classwisetest(model, criterion)
        return best_train_acc, best_test_acc

    def train(self, model, criterion, optimizer):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        model.train()
        
        for step, (inputs, target) in enumerate(self.train_queue):
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),
                                     self.hyperparameters['grad_clip'])
            optimizer.step()
            prec1 = utils.accuracy(logits, target, topk=(1,))
            n = inputs.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1, n)
        return top1.avg, objs.avg

    def infer(self, data,  model, criterion):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        model.eval()
        for step, (inputs, target) in enumerate(data):
            inputs = inputs.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                logits = model(inputs)
                loss = criterion(logits, target)
            prec1 = utils.accuracy(logits, target, topk=(1,))
            n = inputs.size(0)
            objs.update(loss.data.item(), n)
            # top1.update(prec1.data.item(), n)
            top1.update(prec1, n)
            # if step % args.report_freq == 0:
            # logging.info('valid %03d %e %f', step, objs.avg, top1.avg)
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
            # forward pass
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
            # calculate test accuracy for each object class
            for i in range(len(target)):
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
                logging.info('Test Accuracy of %5s: N/A (no training examples)'
                             % (self.classes[i]))
        logging.info('\nTest Accuracy (Overall): %2f%% (%2d/%2d)' % (
          100. * np.sum(class_correct) / np.sum(class_total),
          np.sum(class_correct), np.sum(class_total)))
