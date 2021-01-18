#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : train.py
# @Description : train demo for lanenet
# @IDE: PyCharm Community Edition
# --------------------------------------

import os
import cv2
import time
import math
import argparse
import numpy as np
import tensorflow as tf

from cfg.config import cfg
from data.datasets import Dataset
from model.network import *

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')

    return parser.parse_args()

def train(dataset_dir, weights_path=None):
    """
    :param dataset_dir:
    :param weights_path:
    :return:
    """
    # Set sess configuration
    gpu_options = tf.ConfigProto(allow_soft_placement=True)
    gpu_options.gpu_options.allow_growth = True
    gpu_options.gpu_options.allocator_type = 'BFC'
    config = tf.ConfigProto(gpu_options=gpu_options)

    train_dataset_file = os.path.join(dataset_dir, 'train.txt')
    val_dataset_file = os.path.join(dataset_dir, 'val.txt')

    assert os.path.exists(train_dataset_file)
    assert os.path.exists(val_dataset_file)

    train_dataset = Dataset(train_dataset_file)
    val_dataset = Dataset(val_dataset_file)

    inputs = tf.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.IMG_HEIGHT, cfg.TRAIN.IMG_WIDTH, 3], name='inputs')
    gt_binary_label = tf.placeholder(dtype=tf.int64, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.IMG_HEIGHT, cfg.TRAIN.IMG_WIDTH, 1], name='gt_binary_label')
    gt_instance_label = tf.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.IMG_HEIGHT, cfg.TRAIN.IMG_WIDTH], name='gt_instance_label')

    Model = LaneNet(is_train=True)
    logits = Model.build_model(inputs, 'lanenet')

    losses = Model.compute_loss(logits, binary_label=gt_binary_label, instance_label=gt_instance_label, name='loss')
    total_loss = losses['total_loss']
    segmentation_loss = losses['segmentation_loss']
    discriminative_loss = losses['discriminative_loss']
    pix_embedding = losses['instance_seg_logits']
    seg_logits = losses['binary_seg_logits']

    # calculate the accuracy
    out_logits = tf.nn.softmax(logits=seg_logits)
    out_logits_out = tf.argmax(out_logits, axis=-1)
    out = tf.argmax(out_logits, axis=-1)
    out = tf.expand_dims(out, axis=-1)

    idx = tf.where(tf.equal(binary_label, 1))
    pix_cls_ret = tf.gather_nd(out, idx)
    recall = tf.count_nonzero(pix_cls_ret)
    recall = tf.divide(recall, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))

    idx = tf.where(tf.equal(binary_label, 0))
    pix_cls_ret = tf.gather_nd(out, idx)
    precision = tf.subtract(tf.cast(tf.shape(pix_cls_ret)[0], tf.int64), tf.count_nonzero(pix_cls_ret))
    precision = tf.divide(precision, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))

    accuracy = tf.divide(2.0, tf.divide(1.0, recall) + tf.divide(1.0, precision))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, 100000, 0.1, staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        gradients = optimizer.compute_gradients(total_loss)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, global_step=global_step)

    saver = tf.train.Saver()
    checkpoints_dir = './checkpoints'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    model_save_path = os.path.join(checkpoints_dir, 'lanenet_ckpt')

    logs_save_path = './logs'
    if not os.path.exists(logs_save_path):
        os.makedirs(logs_save_path)
    train_cost_scalar = tf.summary.scalar(name='train_cost', tensor=total_loss)
    val_cost_scalar = tf.summary.scalar(name='val_cost', tensor=total_loss)
    train_accuracy_scalar = tf.summary.scalar(name='train_accuracy', tensor=accuracy)
    val_accuracy_scalar = tf.summary.scalar(name='val_accuracy', tensor=accuracy)
    train_binary_seg_loss_scalar = tf.summary.scalar(name='train_binary_seg_loss', tensor=segmentation_loss)
    val_binary_seg_loss_scalar = tf.summary.scalar(name='val_binary_seg_loss', tensor=segmentation_loss)
    train_instance_seg_loss_scalar = tf.summary.scalar(name='train_instance_seg_loss', tensor=discriminative_loss)
    val_instance_seg_loss_scalar = tf.summary.scalar(name='val_instance_seg_loss', tensor=discriminative_loss)
    learning_rate_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    train_merge_summary_op = tf.summary.merge([train_accuracy_scalar, train_cost_scalar, learning_rate_scalar, train_binary_seg_loss_scalar, train_instance_seg_loss_scalar])
    val_merge_summary_op = tf.summary.merge([val_accuracy_scalar, val_cost_scalar, val_binary_seg_loss_scalar, val_instance_seg_loss_scalar])

    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter(logs_save_path)
    summary_writer.add_graph(sess.graph)

    with sess.as_default():
        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='', name='{:s}/lanenet_model.pbtxt'.format(checkpoints_dir))

        if weights_path is None:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            print('Restore model from last model checkpoint {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        # 加载预训练参数
        # TO DO

        train_cost_time_mean = []
        for epoch in range(cfg.TRAIN.EPOCHS):
            # training part
            t_start = time.time()
            imgs, binary_labels, instance_labels = train_dataset.next_batch(cfg.TRAIN.BATCH_SIZE)
            gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]

            _, c, train_accuracy, train_summary, binary_loss, instance_loss, embedding, binary_seg_img = \
                sess.run([train_op, total_loss,
                          accuracy,
                          train_merge_summary_op,
                          binary_seg_loss,
                          disc_loss,
                          pix_embedding,
                          out_logits_out],
                         feed_dict={input_tensor: gt_imgs,
                                    binary_label_tensor: binary_gt_labels,
                                    instance_label_tensor: instance_gt_labels,
                                    phase: True})

            if math.isnan(c) or math.isnan(binary_loss) or math.isnan(instance_loss):
                log.error('cost is: {:.5f}'.format(c))
                log.error('binary cost is: {:.5f}'.format(binary_loss))
                log.error('instance cost is: {:.5f}'.format(instance_loss))
                log.error('gradients is: {}'.format(g))
                cv2.imwrite('nan_image.png', gt_imgs[0] + VGG_MEAN)
                cv2.imwrite('nan_instance_label.png', instance_gt_labels[0])
                cv2.imwrite('nan_binary_label.png', binary_gt_labels[0] * 255)
                return

            if epoch % 100 == 0:
                cv2.imwrite('image.png', gt_imgs[0] + VGG_MEAN)
                cv2.imwrite('binary_label.png', binary_gt_labels[0] * 255)
                cv2.imwrite('instance_label.png', instance_gt_labels[0])
                cv2.imwrite('binary_seg_img.png', binary_seg_img[0] * 255)

                for i in range(4):
                    embedding[0][:, :, i] = minmax_scale(embedding[0][:, :, i])
                embedding_image = np.array(embedding[0], np.uint8)
                cv2.imwrite('embedding.png', embedding_image)

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            if epoch % cfg.TRAIN.DISPLAY_STEP == 0:
                print('Epoch: {:d} total_loss= {:6f} binary_seg_loss= {:6f} instance_seg_loss= {:6f} accuracy= {:6f} mean_cost_time= {:5f}s '.
                         format(epoch + 1, c, binary_loss, instance_loss, train_accuracy, np.mean(train_cost_time_mean)))
                train_cost_time_mean = []

            if epoch % 1000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

    sess.close()

    return

if __name__ == '__main__':
    args = init_args()
    train(args.dataset_dir, args.weights_path)