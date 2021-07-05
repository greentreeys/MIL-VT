# from torch import *
import torch
import torch.nn.functional as F
import os
import numpy as np
from enum import IntEnum
import cv2
import collections
import threading
import errno
import sys

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(os.path.dirname(fpath))
    torch.save(state, fpath)



class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        count = val.size
        v = val.sum()

        self.count += count
        self.sum += v

        self.avg = self.sum / self.count



def split_by_idxs(seq, idxs):
    '''A generator that returns sequence pieces, seperated by indexes specified in idxs. '''
    last = 0
    for idx in idxs:
        if not (-len(seq) <= idx < len(seq)):
          raise KeyError(f'Idx {idx} is out-of-bounds')
        yield seq[last:idx]
        last = idx
    yield seq[last:]
def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())
def save_model(m, p): torch.save(m.state_dict(), p)
def load_model(m, p):
    sd = torch.load(p, map_location=lambda storage, loc: storage)
    names = set(m.state_dict().keys())
    for n in list(sd.keys()): # list "detatches" the iterator
        if n not in names and n+'_raw' in names and n+'_raw' not in sd:
            sd[n+'_raw'] = sd[n]
            del sd[n]
    m.load_state_dict(sd)

def load_pre(pre, f, fn):
    m = f()
    path = os.path.dirname(__file__)
    if pre: load_model(m, f'{path}/weights/{fn}.pth')
    return m



def to_gpu(x, *args, **kwargs):
    USE_GPU = torch.cuda.is_available()
    '''puts pytorch variable to gpu, if cuda is avaialble and USE_GPU is set to true. '''
    return x.cuda(*args, **kwargs) if USE_GPU else x



def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mean_accuracy(ground_truths, predictions):
    ground_truths = np.array(ground_truths)
    predictions = np.array(predictions)

    class_acc0 = np.sum(
        ground_truths[ground_truths == 0] == predictions[ground_truths == 0]) / np.sum(ground_truths == 0)
    class_acc1 = np.sum(
        ground_truths[ground_truths == 1] == predictions[ground_truths == 1]) / np.sum(ground_truths == 1)
    return class_acc0, class_acc1, (class_acc0+class_acc1) / 2


def multiClassMeanAcc(ground_truths, predictions, class_num):
    ground_truths = np.array(ground_truths)
    predictions = np.array(predictions)

    class_acc = np.zeros(class_num)
    for i in np.arange(class_num):
        class_acc[i] = np.sum(ground_truths[ground_truths == i] == predictions[ground_truths == i]) \
                       / np.sum(ground_truths == i)
    meanAcc = np.mean(class_acc)
    return class_acc, meanAcc

def multiClassPrecision(ground_truths, predictions, class_num):
    
    ground_truths = np.array(ground_truths)
    predictions = np.array(predictions)

    class_precision = np.zeros(class_num)
    for i in np.arange(class_num):
        class_precision[i] = np.sum(ground_truths[ground_truths == i] == predictions[ground_truths == i]) \
                       / np.sum(predictions == i)
    meanPrecision = np.mean(class_precision)
    return class_precision, meanPrecision



def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('denseblock{}'.format(i))
        ft_module_names.append('transition{}'.format(i))
    ft_module_names.append('norm5')
    ft_module_names.append('classifier')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
            else:
                parameters.append({'params': v, 'lr': 0.0})

    return parameters


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def make_weights_for_balanced_classes(DF_train, n_classes):
    nclasses = n_classes
    count = [0] * nclasses
    for i, tempKey in enumerate(range(n_classes)):
        count[i] = np.sum(DF_train['diagnosis'] == tempKey)

    N = float(sum(count))
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(DF_train)
    # classList = [0]*len(DF_train)
    for idx in range(len(DF_train)):
        tempLabel = DF_train.loc[idx, 'diagnosis']
        tempweight = weight_per_class[tempLabel]
        weight[idx] = np.mean(np.array(tempweight))

    return weight


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        print('update lr: ', param_group['lr'])
