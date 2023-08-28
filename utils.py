import os
import yaml
import torch
import shutil
import logging
import numpy as np
from thop import profile
from thop import clever_format
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split
from torchvision.transforms import CenterCrop as CC
from torchvision.transforms import ColorJitter as CJ
from torchvision.transforms import RandomResizedCrop as RRC
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def get_grayscale(dataset):
    grayscale = ['EMNIST', 'FashionMNIST', 'KMNIST']
    if dataset in grayscale:
        return True
    else:
        return False

def get_input_shape(dataset):
    shape = {
             '28' : ["KMNIST", "EMNIST", "FashionMNIST"],
             '32' : ["CIFAR10", "CIFAR100","STL10"],
             '64' : ["EuroSAT"],
             '224' : ["DTD", "Food101", "Flowers102"]
         }
    for key in shape.keys():
        if dataset in shape[key]:
            return int(key)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        # res.append(correct_k.mul_(100.0/batch_size))
        res = correct_k.mul_(100.0/batch_size)
    return res


def log_hash():
    logging.info('#########################################################')
    logging.info('#########################################################\n')


#def _data_transforms_cifar10(args):
#    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
#    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
#    normalize = transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
#    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
#                                          transforms.RandomHorizontalFlip(),
#                                          transforms.ToTensor(),
#                                          normalize])
#    if args.cutout:
#        train_transform.transforms.append(Cutout(args.cutout_length))
#    valid_transform = transforms.Compose([transforms.ToTensor(),
#                                              normalize])
#    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters()
                  if 'auxiliary' not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    print(model_path)
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
        # os.mkdir(os.path.join(path, 'plots'))
        # os.mkdir(os.path.join(path, 'gifs'))
    #print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def load_yaml(yaml_file='config.yaml'):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    return data


def calculate_flops(model, grayscale=False, size=32):
    if grayscale:
        input_tensor = torch.randn(1, 1, size, size).cuda()
    else:
        input_tensor = torch.randn(1, 3, size, size).cuda()
    print('input_tensor = ',input_tensor.shape)
    flops, params = profile(model, inputs=(input_tensor, ))
    flops, params = clever_format([flops, params], "%.3f")
    return flops


def data_load_transforms(cfg):
    dataset_to_run = cfg['dataset_to_run']
    dataset = cfg['datasets'][dataset_to_run]
    #print('\n\ndataset keys =  ', dataset['train'][0])
    input_shape = get_input_shape(dataset_to_run)
    cutout = cfg['flags']['cutout']
    cutout_length = cfg['hyperparameters']['cutout_length']
    # Find mean and std for given dataset
    normalize = transforms.Normalize(mean=dataset['mean'], std=dataset['std'])
    train_transform = transforms.Compose([RRC(input_shape),
                                          transforms.RandomHorizontalFlip(),
                                          CJ(brightness=0.4, contrast=0.4,
                                             saturation=0.4, hue=0.2),
                                          transforms.ToTensor(), normalize])
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))
    test_transform = transforms.Compose([transforms.Resize(input_shape),
                                         CC(input_shape),
                                         transforms.ToTensor(), normalize])
    train_data = eval(dataset['train'][0])
    total_classes = dataset['classes']
    if dataset_to_run == "EuroSAT":     
        test_data = None
        return train_data, test_data, total_classes   
    test_data = eval(dataset['test'][0])
    return train_data, test_data, total_classes
