import torch
import logging
import numpy as np
from PIL import Image
import torch.backends.cudnn as cudnn
import random

def setup_seed(seed):
    random.seed(seed)  # 改变随机生成器的种子
    np.random.seed(seed)  # 用于生成指定随机数
    torch.manual_seed(seed)    # 为CPU设置种子用于生成随机数
    torch.cuda.manual_seed_all(seed)    # 为当前GPU设置随机种子
    # torch.cuda.manual_seed_all(seed)    # 为所有的GPU设置种子
    cudnn.benchmark = True
    cudnn.deterministic = True
    # cudnn.enabled = False

def check(cfg):
    assert cfg.classes > 1
    assert cfg.zoom_factor in [1, 2, 4, 8]
    if cfg.arch == 'psp':
        # assert (cfg.train_h - 1) % 8 == 0 and (cfg.train_w - 1) % 8 == 0
        assert cfg.train_h % 8 == 0 and cfg.train_w % 8 == 0
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    # print(output, target)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)    # each class prediction T
    area_output = torch.histc(output, bins=K, min=0, max=K-1)    # each class prediction
    area_target = torch.histc(target, bins=K, min=0, max=K-1)    # each class target
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger



def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    if args.arch == 'psp':
        # assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
        assert args.train_h % 8 == 0 and args.train_w % 8 == 0
    else:
        raise Exception('architecture not supported yet'.format(args.arch))

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

