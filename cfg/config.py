#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : config.py
# @Description : set global config
# @IDE: PyCharm Community Edition
# --------------------------------------

from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Train options
__C.TRAIN = edict()

# Set training epochs
__C.TRAIN.EPOCHS = 200000
# Set the display step
__C.TRAIN.DISPLAY_STEP = 1
# Set the momentum parameter of the optimizer
__C.TRAIN.MOMENTUM = 0.9
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.0005
# Set the batch size
__C.TRAIN.BATCH_SIZE = 8
# Set the validation batch size
__C.TRAIN.VAL_BATCH_SIZE = 8
# Set the learning rate decay steps
__C.TRAIN.LR_DECAY_STEPS = 400000
# Set the learning rate decay rate
__C.TRAIN.LR_DECAY_RATE = 0.1
# Set the class numbers
__C.TRAIN.CLASSES_NUMS = 2
# Set the image height
__C.TRAIN.IMG_HEIGHT = 256
# Set the image width
__C.TRAIN.IMG_WIDTH = 512

# Test options
__C.TEST = edict()

__C.TEST.GPU_MEMORY_FRACTION = 0.85
