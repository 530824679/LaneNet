#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : datasets.py
# @Description : dataset processor
# @IDE: PyCharm Community Edition
# --------------------------------------

import os
import numpy as np
import cv2
from cfg.config import cfg

class Dataset(object):
    def __init__(self, dataset_file):
        self._img_list, self._label_binary_list, self._label_instance_list = self._init_data_list(dataset_file)
        self._shuffle_dataset()
        self._count = 0

    def _init_data_list(self, dataset_file):
        img_list = []
        label_binary_list = []
        label_instance_list = []

        assert os.path.exists(dataset_file), '{:s}　不存在'.format(dataset_file)

        with open(dataset_file, 'r') as files:
            for file in files:
                info = file.strip(' ').split()
                img_list.append(info[0])
                label_binary_list.append(info[1])
                label_instance_list.append(info[2])

        return img_list, label_binary_list, label_instance_list

    def _shuffle_dataset(self):
        assert len(self._img_list) == len(self._label_binary_list) == len(self._label_instance_list)

        random_idx = np.random.permutation(len(self._img_list))
        shuffle_img_list = []
        shuffle_label_binary_list = []
        shuffle_label_instance_list = []

        for index in random_idx:
            shuffle_img_list.append(self._img_list[index])
            shuffle_label_binary_list.append(self._label_binary_list[index])
            shuffle_label_instance_list.append(self._label_instance_list[index])

        self._img_list = shuffle_img_list
        self._label_binary_list = shuffle_label_binary_list
        self._label_instance_list = shuffle_label_instance_list

    def next_batch(self, batch_size):
        assert len(self._label_binary_list) == len(self._label_instance_list) == len(self._img_list)

        idx_start = batch_size * self._count
        idx_end = batch_size * self._count + batch_size

        if idx_start == 0 and idx_end > len(self._label_binary_list):
            raise ValueError('Batch size not more total samples')

        if idx_end > len(self._label_binary_list):
            self._shuffle_dataset()
            self._count = 0
            return self.next_batch(batch_size)
        else:
            img_list = self._img_list[idx_start:idx_end]
            label_binary_list = self._label_binary_list[idx_start:idx_end]
            label_instance_list = self._label_instance_list[idx_start:idx_end]

            imgs = []
            labels_binary = []
            labels_instance = []

            for img_path in img_list:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (cfg.TRAIN.IMG_WIDTH, cfg.TRAIN.IMG_HEIGHT))
                imgs.append(img)
            for label_path in label_binary_list:
                label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                label_img = label_img / 255
                label_binary = cv2.resize(label_img, (cfg.TRAIN.IMG_WIDTH, cfg.TRAIN.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
                label_binary = np.expand_dims(label_binary, axis=-1)
                labels_binary.append(label_binary)
            for label_path in label_instance_list:
                label_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
                label_img = cv2.resize(label_img, (cfg.TRAIN.IMG_WIDTH, cfg.TRAIN.IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
                labels_instance.append(label_img)

            self._count += 1
            return imgs, labels_binary, labels_instance

if __name__ == '__main__':
    val = Dataset('/training/val.txt')
    b1, b2, b3 = val.next_batch(50)
    c1, c2, c3 = val.next_batch(50)
    dd, d2, d3 = val.next_batch(50)