# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : freeze_graph.py
# Description : 冻结权重ckpt——>pb
# --------------------------------------

import os
import cv2
import numpy as np
from PIL import Image

# 图片后缀
def Postfix():
    postFix = set()
    postFix.update([['bmp', 'jpg', 'png', 'tiff', 'gif', 'pcx', 'tga', 'exif',
                    'fpx', 'svg', 'psd', 'cdr', 'pcd', 'dxf', 'ufo', 'eps', 'JPG', 'raw', 'jpeg'])
    return postFix

# 均值哈希算法
def ahash(image):
    # 将图片缩放到8 * 8
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_CUBIC)
    # 将图片转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 遍历像素累加和, 计算像素平均值
    s = 0
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    avg = s / 64
    # 灰度大于平均值为1，否则为0，得到图片的平均哈希值，此时得到的Hash值为64位的01字符串
    ahash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                ahash_str = ahash_str + '1'
            else:
                ahash_str = ahash_str + '0'
    # 64位转换成16位Hash值
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(ahash_str[i: i + 4], 2))
    print("ahash值：",result)
    return result

# 感知哈希算法
def phash(image):
    # 将图片缩放到32 * 32
    image_resize = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
    # 将图片转换为灰度图
    gray = cv2.cvtColor(image_resize, cv2.COLOR_RGB2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    img_dct = cv2.dct(np.float32(gray))
    # 获取左上角8 * 8 的ROI区域
    roi_dct = img_dct[0:8, 0:8]
    # 计算均值
    avreage = np.mean(roi_dct)
    # 计算哈希值
    hash_list = []
    for i in range(roi_dct.shape[0]):
        for j in range(roi_dct.shape[1]):
            if roi_dct[i, j] > avreage:
                hash_list.append(1)
            else:
                hash_list.append(0)
    # 64位转换成16位Hash值
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(''.join(hash_list[i:i + 4]), 2))
    print("phash值：", result)
    return result

# 差异哈希算法
def dhash(image):
    # 将图片缩放到9 * 8
    image_resize = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 将图片转换为灰度图
    gray = cv2.cvtColor(image_resize, cv2.COLOR_RGB2GRAY)
    # 每行前一个像素大于后一个像素为1，否则为0，得到图片的平均哈希值，此时得到的Hash值为64位的01字符串
    dhash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                ahash_str = dhash_str + '1'
            else:
                ahash_str = dhash_str + '0'
    # 64位转换成16位Hash值
    result = ''
    for i in range(0, 64, 4):
        result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
    # print("dhash值",result)
    return result

# 计算两张图之间的汉明距离
def Hamming(hash1, hash2):
    # hash长度不同返回-1,此时不能比较
    if len(hash1) != len(hash2):
        return -1

    hamming_distance = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            hamming_distance += 1
    return hamming_distance

# 图片归一化
def Normalize(image, size=(64, 64), greyscale=False):
    # 重新设置图片大小
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image

# 计算两张图之间的余弦距离
def Cosine(image1, image2):
    image1 = Normalize(image1)
    image2 = Normalize(image2)

    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(np.average(pixel_tuple))
        vectors.append(vector)
        norms.append(np.linalg.norm(vector, 2))

    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = np.dot(a / a_norm, b / b_norm)
    return res

def Histogram(image_1, image_2):
    # 计算单通道直方图
    hist_1 = cv2.calcHist([image_1], [0], None, [256], [0.0, 255.0])
    hist_2 = cv2.calcHist([image_2], [0], None, [256], [0.0, 255.0])

    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist_1)):
        if hist_1[i] != hist_2[i]:
            degree = degree + (1 - abs(hist_1[i] - hist_2[i]) / max(hist_1[i], hist_2[i]))
        else:
            degree = degree + 1

    degree = degree / len(hist_1)
    return degree

if __name__ == '__main__':
    threshold = 100
    image_dir = ""

    postFix = Postfix()
    file_list = os.listdir(image_dir)
    for file in file_list:
        if str(file).split('.')[-1] in postFix:
            image = Image.open(r'%s/%s' % (image_dir, str(file)))
            diff =




