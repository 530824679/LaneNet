#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : network.py
# @Description : network architecture
# @IDE: PyCharm Community Edition
# --------------------------------------

import tensorflow as tf
from model.enet import *
from model.base import *

class LaneNet(BaseModel):
    def __init__(self, is_train):
        super(LaneNet, self).__init__()
        self.is_train = is_train

    def build_model(self, inputs, name):
        with tf.variable_scope(name):
            '''
            LaneNet only shares the first two stages between the two branches, leaving stage 3 of the ENet encoder
            and the full ENet decoder as the backbone of each separate branch.
            '''
            # shared stages
            with tf.variable_scope('LaneNetBase'):
                initial = iniatial_block(inputs, isTraining=self.is_train)
                stage1, pooling_indices_1, inputs_shape_1 = ENet_stage1(initial, isTraining=self.is_train)
                stage2, pooling_indices_2, inputs_shape_2 = ENet_stage2(stage1, isTraining=self.is_train)

            # Segmentation branch
            with tf.variable_scope('LaneNetSeg'):
                segStage3 = ENet_stage3(stage2, isTraining=self.is_train)
                segStage4 = ENet_stage4(segStage3, pooling_indices_2, inputs_shape_2, stage1, isTraining=self.is_train)
                segStage5 = ENet_stage5(segStage4, pooling_indices_1, inputs_shape_1, initial, isTraining=self.is_train)
                segLogits = tf.layers.conv2d_transpose(segStage5, 2, [2, 2], strides=2, padding='same', name='fullconv')

            # Embedding branch
            with tf.variable_scope('LaneNetEm'):
                emStage3 = ENet_stage3(stage2, isTraining=self.is_train)
                emStage4 = ENet_stage4(emStage3, pooling_indices_2, inputs_shape_2, stage1, isTraining=self.is_train)
                emStage5 = ENet_stage5(emStage4, pooling_indices_1, inputs_shape_1, initial, isTraining=self.is_train)
                emLogits = tf.layers.conv2d_transpose(emStage5, 4, [2, 2], strides=2, padding='same', name='fullconv')

            predicts = {
                'logits': segLogits,
                'deconv': emLogits
            }

            return predicts

    def inference(self, inputs, name):
        with tf.variable_scope(name):
            logits = self.build_model(inputs, name='inference')

            # 计算图像分割结果
            decode_logits = logits['logits']
            binary_seg = tf.nn.softmax(logits=decode_logits)
            binary_seg = tf.argmax(binary_seg, axis=-1)

            # 计算像素嵌入结果
            decode_deconv = logits['deconv']
            pix_embedding = self.conv2d(inputdata=decode_deconv, out_channel=4, kernel_size=1, use_bias=False, name='pix_embedding_conv')
            pix_embedding = self.relu(inputdata=pix_embedding, name='pix_embedding_relu')

            return binary_seg, pix_embedding

    def compute_loss(self, predicts, binary_label, instance_label, name):
        with tf.variable_scope(name):
            decode_logits = predicts['logits']
            decode_deconv = predicts['deconv']

            # 计算二值分割损失函数
            binary_label_plain = tf.reshape(
                binary_label,
                shape=[binary_label.get_shape().as_list()[0] *
                       binary_label.get_shape().as_list()[1] *
                       binary_label.get_shape().as_list()[2]])
            # 加入class weights
            unique_labels, unique_id, counts = tf.unique_with_counts(binary_label_plain)
            counts = tf.cast(counts, tf.float32)
            inverse_weights = tf.divide(1.0, tf.log(tf.add(tf.divide(tf.constant(1.0), counts), tf.constant(1.02))))
            inverse_weights = tf.concat([tf.constant([5.]), inverse_weights[1:]], axis=0)
            inverse_weights = tf.gather(inverse_weights, binary_label)

            segmenatation_loss = tf.losses.sparse_softmax_cross_entropy(labels=binary_label, logits=decode_logits, weights=inverse_weights)
            segmenatation_loss = tf.reduce_mean(segmenatation_loss)

            # 计算像素区别损失函数
            pix_embedding = self.conv2d(inputdata=decode_deconv, out_channel=4, kernel_size=1, use_bias=False, name='pix_embedding_conv')
            pix_embedding = self.relu(inputdata=pix_embedding, name='pix_embedding_relu')

            image_shape = (pix_embedding.get_shape().as_list()[1], pix_embedding.get_shape().as_list()[2])
            discriminative_loss, l_var, l_dist, l_reg = discriminative_loss.discriminative_loss(pix_embedding, instance_label, 4, image_shape, 0.5, 3.0, 1.0, 1.0, 0.001)

            # 合并损失
            l2_reg_loss = tf.constant(0.0, tf.float32)
            for vv in tf.trainable_variables():
                if 'bn' in vv.name or 'batchnorm' in vv.name or 'batch_norm' in vv.name and 'alpha' in vv.name:
                    continue
                else:
                    l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
            l2_reg_loss *= 0.001
            total_loss = 0.5 * segmenatation_loss + 0.5 * discriminative_loss + l2_reg_loss

            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': decode_logits,
                'instance_seg_logits': pix_embedding,
                'binary_seg_loss': segmenatation_loss,
                'discriminative_loss': discriminative_loss
            }

            return ret

if __name__ == '__main__':
    input = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    binary_label = tf.placeholder(dtype=tf.int64, shape=[1, 256, 512, 1], name='label')
    instance_label = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 1], name='label')

    model = LaneNet(tf.constant('train', dtype=tf.string))
    logits = model.build_model(input, 'lanenet')
    ret = model.compute_loss(logits, binary_label=binary_label, instance_label=instance_label, name='loss')
    for vv in tf.trainable_variables():
        if 'bn' in vv.name:
            continue
        print(vv.name)