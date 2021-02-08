# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : freeze_graph.py
# Description : 冻结权重ckpt——>pb
# --------------------------------------

import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
import numpy as np
from PIL import Image

# 图片后缀
def Postfix():
    postFix = set()
    postFix.update(['bmp', 'jpg', 'png', 'tiff', 'gif', 'pcx', 'tga', 'exif',
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
    # print("ahash值：",ahash_str)
    return ahash_str

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
    phash_str = ''
    for i in range(roi_dct.shape[0]):
        for j in range(roi_dct.shape[1]):
            if roi_dct[i, j] > avreage:
                phash_str = phash_str + '1'
            else:
                phash_str = phash_str + '0'
    # print("phash值：",phash_str)
    return phash_str

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
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    # print("dhash值", dhash_str)
    return dhash_str

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
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    if greyscale:
        # 将图片转换为灰度图，其每个像素用8个bit表示
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

# 计算两张图之间的余弦距离
def Cosine(image1, image2):
    image1 = Normalize(image1)
    image2 = Normalize(image2)

    image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

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

def correlation(image, kernal):
    kernal_heigh = kernal.shape[0]
    kernal_width = kernal.shape[1]
    cor_heigh = image.shape[0] - kernal_heigh + 1
    cor_width = image.shape[1] - kernal_width + 1
    result = np.zeros((cor_heigh, cor_width), dtype=np.float64)
    for i in range(cor_heigh):
        for j in range(cor_width):
            result[i][j] = (image[i:i + kernal_heigh, j:j + kernal_width] * kernal).sum()
    return result

def gaussian_2d_kernel(kernel_size=11, sigma=1.5):
    kernel = np.zeros([kernel_size, kernel_size])
    center = kernel_size // 2

    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
    sum_val = 1 / sum_val
    return kernel * sum_val

def ssim(image_1, image_2, window_size=11, gaussian_sigma=1.5, K1=0.01, K2=0.03, alfa=1, beta=1, gama=1):
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

    image_1=np.array(image_1,dtype=np.float64)
    image_2=np.array(image_2,dtype=np.float64)

    if not image_1.shape == image_2.shape:
        raise ValueError("Input Imagees must has the same size")

    if len(image_1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    kernal=gaussian_2d_kernel(window_size,gaussian_sigma)

    # 求ux uy ux*uy ux^2 uy^2 sigma_x^2 sigma_y^2 sigma_xy等中间变量
    ux = correlation(image_1, kernal)
    uy = correlation(image_2, kernal)
    image_1_sqr = image_1 ** 2
    image_2_sqr = image_2 ** 2
    dis_mult_ori = image_1 * image_2

    uxx = correlation(image_1_sqr, kernal)
    uyy = correlation(image_2_sqr, kernal)
    uxy = correlation(dis_mult_ori, kernal)
    ux_sqr = ux ** 2
    uy_sqr = uy ** 2
    uxuy = ux * uy
    sx_sqr = uxx - ux_sqr
    sy_sqr = uyy - uy_sqr
    sxy = uxy - uxuy
    C1 = (K1 * 255) ** 2
    C2 = (K2 * 255) ** 2

    #常用情况的SSIM
    if(alfa==1 and beta==1 and gama==1):
        ssim=(2 * uxuy + C1) * (2 * sxy + C2) / (ux_sqr + uy_sqr + C1) / (sx_sqr + sy_sqr + C2)
        return np.mean(ssim)

    #计算亮度相似性
    l = (2 * uxuy + C1) / (ux_sqr + uy_sqr + C1)
    l = l ** alfa

    #计算对比度相似性
    sxsy = np.sqrt(sx_sqr) * np.sqrt(sy_sqr)
    c= (2 * sxsy + C2) / (sx_sqr + sy_sqr + C2)
    c= c ** beta

    #计算结构相似性
    C3 = 0.5 * C2
    s = (sxy + C3) / (sxsy + C3)
    s = s ** gama

    ssim = l * c * s
    return np.mean(ssim)

if __name__ == '__main__':
    image_dir = "/home/chenwei/HDD/Project/datasets/segmentation/lane_dataset/高速通道/白天/image"

    postFix = Postfix()
    file_list = os.listdir(image_dir)
    file_list.sort()

    cImage = cv2.imread(r'%s/%s' % (image_dir, str(file_list[0])))
    sp = cImage.shape
    top = int(sp[0] / 2)
    bottom = int(sp[0])
    cROI = cImage[:, top:bottom]
    cValue = dhash(cROI)

    count = 0
    for index in range(1, len(file_list)):
        if str(file_list[index]).split('.')[-1] in postFix:
            image = cv2.imread(r'%s/%s' % (image_dir, str(file_list[index])))
            roi = image[:, top:bottom]
            value = dhash(roi)

            #dis = Hamming(cValue, value)
            dis = ssim(cROI, roi)
            print("Distance is: ", dis)

            concat = np.hstack((cImage, image))
            concat = cv2.resize(concat, (960, 360), interpolation=cv2.INTER_CUBIC)
            cv2.line(concat, (0, 180), (960, 180), (0, 255, 0), 1, 4)

            cv2.imshow("test", concat)
            cv2.waitKey(75)
            if dis < 0.6:
                cv2.imwrite('/home/chenwei/HDD/Project/datasets/segmentation/lane_dataset/result/' + str(count) + '.png', cImage)
                cImage = image
                #cValue = value
                cROI = roi
                count += 1

