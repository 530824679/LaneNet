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

import os

path_params = {
    'data_path': "/home/chenwei/HDD/Project/datasets/segmentation/tusimple",
    'test_files_list': 'REPO_ROOT_PATH/data/training_data_example/test.txt',
    'train_files_list': 'REPO_ROOT_PATH/data/training_data_example/train.txt',
    'val_files_list': 'REPO_ROOT_PATH/data/training_data_example/val.txt',
    'logs_dir': './logs',
    'tfrecord_dir': './tfrecord',
    'checkpoints_dir': './checkpoints',
}

model_params = {
    'resize_image_size': [720, 720],
    'train_image_size': [512, 256],
    'channels': 3,
}

solver_params = {
    'batch_size': 8,
    'total_epoches': 500,
}

test_params = {

}
