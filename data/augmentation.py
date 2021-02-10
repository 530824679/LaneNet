# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : augmentation.py
# Description : data augmentation
# --------------------------------------

import glog as log
import tensorflow as tf

def central_crop(image, crop_height, crop_width):
    """
    Performs central crops of the given image
    :param image:
    :param crop_height:
    :param crop_width:
    :return:
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    cropped_h = (height - crop_height)
    crop_top = cropped_h // 2
    cropped_w = (width - crop_width)
    crop_left = cropped_w // 2

    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

def random_color_augmentation(gt_image, gt_binary_image, gt_instance_image):
    """
    andom color augmentation
    :param gt_image:
    :param gt_binary_image:
    :param gt_instance_image:
    :return:
    """
    # first apply random saturation augmentation
    gt_image = tf.image.random_saturation(gt_image, 0.8, 1.2)
    # sencond apply random brightness augmentation
    gt_image = tf.image.random_brightness(gt_image, 0.05)
    # third apply random contrast augmentation
    gt_image = tf.image.random_contrast(gt_image, 0.7, 1.3)

    gt_image = tf.clip_by_value(gt_image, 0.0, 255.0)

    return gt_image, gt_binary_image, gt_instance_image

def random_horizon_flip_images(gt_image, gt_binary_image, gt_instance_image):
    """
    Random horizon flip image data for training
    :param gt_image:
    :param gt_binary_image:
    :param gt_instance_image:
    :return:
    """
    concat_images = tf.concat([gt_image, gt_binary_image, gt_instance_image], axis=-1)

    [image_height, image_width, _] = gt_image.get_shape().as_list()

    concat_flipped_images = tf.image.random_flip_left_right(image=concat_images, seed=tf.random.set_random_seed(1))

    flipped_gt_image = tf.slice(concat_flipped_images, begin=[0, 0, 0], size=[image_height, image_width, 3])
    flipped_gt_binary_image = tf.slice(concat_flipped_images, begin=[0, 0, 3], size=[image_height, image_width, 1])
    flipped_gt_instance_image = tf.slice(concat_flipped_images, begin=[0, 0, 4], size=[image_height, image_width, 1])

    return flipped_gt_image, flipped_gt_binary_image, flipped_gt_instance_image

def random_crop_images(gt_image, gt_binary_image, gt_instance_image, cropped_size):
    """
    Random crop image batch data for training
    :param gt_image:
    :param gt_binary_image:
    :param gt_instance_image:
    :param cropped_size:
    :return:
    """
    concat_images = tf.concat([gt_image, gt_binary_image, gt_instance_image], axis=-1)

    concat_cropped_images = tf.image.random_crop(concat_images, [cropped_size[1], cropped_size[0], tf.shape(concat_images)[-1]], seed=tf.random.set_random_seed(1234))

    cropped_gt_image = tf.slice(concat_cropped_images, begin=[0, 0, 0], size=[cropped_size[1], cropped_size[0], 3])
    cropped_gt_binary_image = tf.slice(concat_cropped_images, begin=[0, 0, 3], size=[cropped_size[1], cropped_size[0], 1])
    cropped_gt_instance_image = tf.slice(concat_cropped_images, begin=[0, 0, 4], size=[cropped_size[1], cropped_size[0], 1])

    return cropped_gt_image, cropped_gt_binary_image, cropped_gt_instance_image

def augment_for_train(gt_image, gt_binary_image, gt_instance_image):
    """
    :param gt_image:
    :param gt_binary_image:
    :param gt_instance_image:
    :return:
    """
    # convert image from uint8 to float32
    gt_image = tf.cast(gt_image, tf.float32)
    gt_binary_image = tf.cast(gt_binary_image, tf.float32)
    gt_instance_image = tf.cast(gt_instance_image, tf.float32)

    # apply random color augmentation
    gt_image, gt_binary_image, gt_instance_image = random_color_augmentation(gt_image, gt_binary_image, gt_instance_image)

    # apply random flip augmentation
    gt_image, gt_binary_image, gt_instance_image = random_horizon_flip_images(gt_image, gt_binary_image, gt_instance_image)

    # apply random crop image
    gt_image, gt_binary_image, gt_instance_image = random_crop_images(gt_image=gt_image, gt_binary_image=gt_binary_image, gt_instance_image=gt_instance_image, cropped_size=[CROP_IMAGE_WIDTH, CROP_IMAGE_HEIGHT])

    return gt_image, gt_binary_image, gt_instance_image


def augment_for_test(gt_image, gt_binary_image, gt_instance_image):
    """

    :param gt_image:
    :param gt_binary_image:
    :param gt_instance_image:
    :return:
    """
    # apply central crop
    gt_image = central_crop(image=gt_image, crop_height=CROP_IMAGE_HEIGHT, crop_width=CROP_IMAGE_WIDTH)
    gt_binary_image = central_crop(image=gt_binary_image, crop_height=CROP_IMAGE_HEIGHT, crop_width=CROP_IMAGE_WIDTH)
    gt_instance_image = central_crop(image=gt_instance_image, crop_height=CROP_IMAGE_HEIGHT, crop_width=CROP_IMAGE_WIDTH)

    return gt_image, gt_binary_image, gt_instance_image


def normalize(gt_image, gt_binary_image, gt_instance_image):
    """
    Normalize the image data by substracting the imagenet mean value
    :param gt_image:
    :param gt_binary_image:
    :param gt_instance_image:
    :return:
    """

    if gt_image.get_shape().as_list()[-1] != 3 or gt_binary_image.get_shape().as_list()[-1] != 1 or gt_instance_image.get_shape().as_list()[-1] != 1:
        log.error(gt_image.get_shape())
        log.error(gt_binary_image.get_shape())
        log.error(gt_instance_image.get_shape())
        raise ValueError('Input must be of size [height, width, C>0]')

    gt_image = tf.cast(gt_image, dtype=tf.float32)
    gt_image = tf.subtract(tf.divide(gt_image, tf.constant(127.5, dtype=tf.float32)), tf.constant(1.0, dtype=tf.float32))

    return gt_image, gt_binary_image, gt_instance_image