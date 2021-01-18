#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : test.py
# @Description : test demo for lanenet
# @IDE: PyCharm Community Edition
# --------------------------------------

import os
import time
import math
import argparse
import numpy as np
import tensorflow as tf
from model.network import *
from cfg.config import *
import cv2

def test_single_image(image_path, weights_path):
    assert os.path.exists(image_path), '{:s} not exist'.format(image_path)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 255.0

    inputs = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='inputs')
    model = LaneNet(is_train=False)
    binary_seg, instance_seg = model.inference(inputs, name='lanenet')


    saver = tf.train.Saver()
    config = tf.ConfigProto(device_count={'GPU': 0})
    config.gpu_options.per_process_gpu_memory_fraction = cfg.TEST.GPU_MEMORY_FRACTION
    config.gpu_options.allow_growth = cfg.TRAIN.TF_ALLOW_GROWTH
    config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        binary_seg_image, instance_seg_image = sess.run([binary_seg, instance_seg], feed_dict={inputs: [image]})
        t_cost = time.time() - t_start
        print('单张图像车道线预测耗时: {:.5f}s'.format(t_cost))

        # 删除一些比较小的联通区域
        # binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])
        t_start = time.time()
        mask_image, _, _, _ = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0], instance_seg_ret=instance_seg_image[0])
        t_cost = time.time() - t_start
        print('单张图像车道线聚类耗时: {:.5f}s'.format(t_cost))

        print(instance_seg_image.shape)
        for i in range(4):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        cv2.imwrite('./out/out_bin_img.png', binary_seg_image[0] * 255)
        cv2.imwrite('./out/out_mask_img.png', mask_image)
        cv2.imwrite('./out/out_ori_img.png', image_vis)
        cv2.imwrite('./out/out_ins_img.png', embedding_image)