# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : freeze_graph.py
# Description : 冻结权重ckpt——>pb
# --------------------------------------

import os
import numpy as np
import tensorflow as tf

with tf.device('/cpu:0'):
    pb_path = './checkpoints/model.pb'
    ckpt_path = './checkpoints/model.ckpt'
    meta_path = './checkpoints/model.meta'
    output_node_names = ["lanenet/LaneNetSeg/fullconv/conv2d_transpose"]

    with tf.Session() as sess:
        # Restore the graph
        saver = tf.train.import_meta_graph(meta_path)

        # Load weights
        saver.restore(sess, ckpt_path)

        # Freeze the graph
        converted_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def=sess.graph, output_node_names=output_node_names)

        # Save the frozen graph
        with tf.gfile.GFile(pb_path, "wb") as f:
            f.write(converted_graph_def.SerializeToString())

        print("=> {} ops written to {}.".format(len(converted_graph_def.node), pb_path))