# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : tfrecord.py
# Description :create and parse tfrecord
# --------------------------------------

import os
import cv2
import glog as log
import numpy as np
import tensorflow as tf

from cfg.config import *

class TFRecord(object):
    def __init__(self):
        self.data_path = path_params['data_path']
        self.tfrecord_dir = path_params['tfrecord_dir']
        self.channels = model_params['channels']
        self.resize_image_size = model_params['resize_image_size']
        self.train_image_size = model_params['train_image_size']
        self.batch_size = solver_params['batch_size']

    def int64_feature(self, value):
        """

        :return:
        """
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def bytes_feature(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def write_example_tfrecords(self, gt_images_paths, gt_binary_images_paths, gt_instance_images_paths, tfrecords_path):
        """
        write tfrecords
        :param gt_images_paths:
        :param gt_binary_images_paths:
        :param gt_instance_images_paths:
        :param tfrecords_path:
        :return:
        """
        tfrecords_dir = os.split(tfrecords_path)[0]
        os.makedirs(tfrecords_dir, exist_ok=True)

        log.info('Writing {:s}....'.format(tfrecords_path))
        with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
            for index, gt_image_path in enumerate(gt_images_paths):

                # prepare gt image
                gt_image = cv2.imread(gt_image_path, cv2.IMREAD_UNCHANGED)
                if gt_image.shape != (self.train_image_size[0], self.train_image_size[1], self.channels):
                    gt_image = cv2.resize(gt_image, dsize=(self.train_image_size[0], self.train_image_size[1]), interpolation=cv2.INTER_LINEAR)
                gt_image_raw = gt_image.tostring()

                # prepare gt binary image
                gt_binary_image = cv2.imread(gt_binary_images_paths[index], cv2.IMREAD_UNCHANGED)
                if gt_binary_image.shape != (self.train_image_size[0], self.train_image_size[1]):
                    gt_binary_image = cv2.resize(gt_binary_image, dsize=(self.train_image_size[0], self.train_image_size[1]), interpolation=cv2.INTER_NEAREST)
                    gt_binary_image = np.array(gt_binary_image / 255.0, dtype=np.uint8)
                gt_binary_image_raw = gt_binary_image.tostring()

                # prepare gt instance image
                gt_instance_image = cv2.imread(gt_instance_images_paths[index], cv2.IMREAD_UNCHANGED)
                if gt_instance_image.shape != (self.train_image_size[0], self.train_image_size[1]):
                    gt_instance_image = cv2.resize(gt_instance_image, dsize=(self.train_image_size[0], self.train_image_size[1]), interpolation=cv2.INTER_NEAREST)
                gt_instance_image_raw = gt_instance_image.tostring()

                example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'gt_image_raw': self.bytes_feature(gt_image_raw),
                            'gt_binary_image_raw': self.bytes_feature(gt_binary_image_raw),
                            'gt_instance_image_raw': self.bytes_feature(gt_instance_image_raw)
                        }))
                writer.write(example.SerializeToString())
        writer.close()
        log.info('Writing {:s} complete'.format(tfrecords_path))

        return

    def parse_single_example(self, serialized_example):
        """
        Parses an image and label from the given `serialized_example`
        :param file_name:待解析的tfrecord文件的名称
        :return: 从文件中解析出的单个样本的相关特征，image, label
        """
        features = tf.parse_single_example(
            serialized_example,
            features={
                'gt_image_raw': tf.FixedLenFeature([], tf.string),
                'gt_binary_image_raw': tf.FixedLenFeature([], tf.string),
                'gt_instance_image_raw': tf.FixedLenFeature([], tf.string)
            })

        # decode gt image
        gt_image_shape = tf.stack([self.train_image_size[1], self.train_image_size[0], 3])
        gt_image = tf.decode_raw(features['gt_image_raw'], tf.uint8)
        gt_image = tf.reshape(gt_image, gt_image_shape)

        # decode gt binary image
        gt_binary_image_shape = tf.stack([self.train_image_size[1], self.train_image_size[0], 1])
        gt_binary_image = tf.decode_raw(features['gt_binary_image_raw'], tf.uint8)
        gt_binary_image = tf.reshape(gt_binary_image, gt_binary_image_shape)

        # decode gt instance image
        gt_instance_image_shape = tf.stack([self.train_image_size[1], self.train_image_size[0], 1])
        gt_instance_image = tf.decode_raw(features['gt_instance_image_raw'], tf.uint8)
        gt_instance_image = tf.reshape(gt_instance_image, gt_instance_image_shape)

        return gt_image, gt_binary_image, gt_instance_image