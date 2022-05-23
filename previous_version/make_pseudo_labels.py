# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.puzzle_utils import *
from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--experiment_name', default='', type=str)
parser.add_argument('--domain', default='train_aug', type=str)

parser.add_argument('--threshold', default=0.25, type=float)
parser.add_argument('--crf_iteration', default=0, type=int)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    from iputils import get_host_ip

    ip = get_host_ip()
    if ip == '172.31.234.159':
        args.data_dir = '/data1/xjheng/dataset/VOC2012/'
    elif ip == '172.31.111.180':
        args.data_dir = '/home/lzr/data/VOC/VOC2012/'
    else:
        raise NotImplementedError
    cam_dir = f'./experiments/predictions/{args.experiment_name}/'
    pred_dir = create_directory(f'./experiments/predictions/{args.experiment_name}@crf={args.crf_iteration}/')

    set_seed(args.seed)
    log_func = lambda string='': print(string)
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, label, gt_mask) in enumerate(dataset):
            png_path = pred_dir + image_id + '.png'
            if os.path.isfile(png_path):
                continue
            
            ori_w, ori_h = ori_image.size
            predict_dict = np.load(cam_dir + image_id + '.npy', allow_pickle=True).item()

            keys = predict_dict['keys']
            
            cams = predict_dict['rw']
            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)

            cams = np.argmax(cams, axis=0)

            if args.crf_iteration > 0:
                cams = crf_inference_label(np.asarray(ori_image), cams, n_labels=keys.shape[0], t=args.crf_iteration)
            
            conf = keys[cams]
            imageio.imwrite(png_path, conf.astype(np.uint8))
            
            # cv2.imshow('image', np.asarray(ori_image))
            # cv2.imshow('predict', decode_from_colormap(predict, dataset.colors))
            # cv2.waitKey(0)

            sys.stdout.write('\r# Make Pseudo Labels [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), conf.shape))
            sys.stdout.flush()
        print()
    
    print("python3 evaluate.py --experiment_name {} --mode png".format(args.experiment_name + f'@crf={args.crf_iteration}'))
