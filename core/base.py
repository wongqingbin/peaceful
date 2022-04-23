#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Dict

import requests

import cv2
import numpy as np


class PyImage:

    def __init__(self, strategy: Dict = None):
        strategy = strategy if strategy else {
            'gray': 0,
            'thresh': 0,  # 127
            'noise': 0
        }
        super(PyImage, self).__init__()

    def read(self, path: str) -> np.ndarray:
        """
        OpenCV读取图片，URL或本地路径读取
        :param path: URL or image_path
        :return: array矩阵: matrix_BGR[x][y] = [B,G,R]
        """
        if path.startswith('http://') or path.startswith('https://'):
            resp = requests.get(url=path).content
            image_array = np.asarray(bytearray(resp), dtype="uint8")
            BGR_array = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        else:
            # 将图像调整为3通道的BGR图像，该值为默认值。cv2.IMREAD_COLOR == 1
            BGR_array = cv2.imread(path, cv2.IMREAD_COLOR)

        if not BGR_array.all():  # 多维数组判空
            print("图片读取失败")
        # cv2.namedWindow('demo')
        # cv2.imshow('demo', BGR_array)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return BGR_array

    def write(self, path: str, image: np.ndarray, params=None) -> bool:
        """
        保存文件
        :param params:
        :param image:
        :param path:
        :return:
        """
        # 上传到URL
        img_encode = cv2.imencode('.png', image)[1]
        bytes_encode = np.array(img_encode).tobytes()
        files = {'file': bytes_encode}
        requests.request(method='', url='', files=files)

        # 保存至本地
        flag = cv2.imwrite(filename=path, img=image, params=params)
        return flag

    def get_info(self, image: np.ndarray):
        # height, width, depth = image.shape  # 高 宽 深(通道数)
        # size = image.size  # 图片大小
        return image.shape

    def to_gray(self, image: np.ndarray) -> np.ndarray:
        """
        cv2.COLOR_RGB2GRAY -->  BGR->GRAY
        :param image:
        :return:
        """
        # 默认算法
        gray_array = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 平均值法
        # h, w = image.shape[:2]
        # gray_average = np.zeros((h, w), dtype=np.uint8)
        # for i in range(h):
        #     for j in range(w):
        #         b = int(bgr_array[i, j][0])
        #         g = int(bgr_array[i, j][1])
        #         r = int(bgr_array[i, j][2])
        #         gray_average[i, j] = round((b + g + r) / 3)

        # 最大值法
        # h, w = image.shape[:2]
        # gray_max = np.zeros((h, w), dtype=np.uint8)
        # for i in range(h):
        #     for j in range(w):
        #         b = int(bgr_array[i, j][0])
        #         g = int(bgr_array[i, j][1])
        #         r = int(bgr_array[i, j][2])
        #         gray_max[i, j] = max(b, g, r)

        # 分量法
        # for i in range(image.shape[0]):
        #     for j in range(image.shape[1]):
        #         image[i, j] = image[i, j, 0]

        # 加权平均法 Y = 0．3R + 0．59G + 0．11B
        # h, w = image.shape[:2]
        # gray_weighted = np.zeros((h, w), dtype=np.uint8)
        # for i in range(h):
        #     for j in range(w):
        #         # 通过cv格式打开的图片，像素格式为 BGR
        #         b = int(image[i, j][0])
        #         g = int(image[i, j][1])
        #         r = int(image[i, j][2])
        #         gray_weighted[i, j] = 0.3 * r + 0.11 * g + 0.59 * b

        return gray_array

    def to_binary(self, image: np.ndarray) -> np.ndarray:
        """

        :param image:
        :return:
        """
        # 固定阈值 手动决定thresh大小
        # thresh = 127
        # thresh, binary_array = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)

        # 自适应阈值 thresh由自适应方法动态得出,一张灰度图中可以分成多个区域阈值
        # binary_array1 = cv2.adaptiveThreshold(  # 自适应高斯加权
        #     src=image,
        #     maxValue=255,  # 像素最大值
        #     adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 表示每个区域块的阈值,由区域块内每个像素的灰度值经过高斯函数加权相乘后再相加，最后减去常量C得出
        #     thresholdType=cv2.THRESH_BINARY,  # thresholdType
        #     blockSize=5,  # 区域块大小(必须是大于1的奇数) 通常为3、5、7
        #     C=3  # 常数C(偏移量,用于矫正阈值)
        # )
        # binary_array2 = cv2.adaptiveThreshold(  # 自适应平均 缺点是可能会忽略掉区域内某些特殊的像素值=>例如照片中衣服上的花纹
        #     src=image,
        #     maxValue=255,  # 像素最大值
        #     adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,  # 表示每个区域块的阈值,由区域块内每个像素的灰度值经过高斯函数加权相乘后再相加，最后减去常量C得出
        #     thresholdType=cv2.THRESH_BINARY,  # thresholdType
        #     blockSize=5,  # 区域块大小(必须是大于1的奇数) 通常为3、5、7
        #     C=3  # 常数C(偏移量,用于矫正阈值)
        # )

        # Otsu阈值 自动获取全局阈值,即无需手动决定阈值大小,阈值thresh大小默认0即可
        thresh, binary_array = cv2.threshold(
            src=image,
            thresh=0,  # OTSU自动获取阈值时，此值必须填0
            maxval=255,
            type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # thresh, binary_array2 = cv2.threshold(
        #     src=image,
        #     thresh=0,  # OTSU自动获取阈值时，此值必须填0
        #     maxval=255,
        #     type=cv2.THRESH_TOZERO + cv2.THRESH_OTSU
        # )

        return binary_array

    def xor(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """

        :param image1:
        :param image2:
        :return:
        """
        results = cv2.bitwise_xor(image1, image2)
        # cv2.imshow('bitwise_xor', results)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return results

    def denoising(self, image: np.ndarray) -> np.ndarray:
        """
        卷积核默认为 (5,5)
        :param image:
        :return:
        """
        ksize = (5, 5)

        # 均值滤波
        # image = cv2.blur(image, ksize=ksize)

        # 高斯模糊绿波
        # image = cv2.GaussianBlur(image, ksize, 0, 0)

        # 中值滤波
        image = cv2.medianBlur(image, 5)

        # 双边滤波
        # image = cv2.bilateralFilter(image, 5, 100, 100)

        # 方框滤波
        # image = cv2.boxFilter(image, -1, ksize=ksize)

        # 2D卷积 自定义卷积核进行均值滤波
        # kernel = np.ones((9, 9), np.float32) / 81
        # image = cv2.filter2D(image, -1, kernel=kernel)

        # cv2.imshow('denoising', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return image

    def draw_box(self, image: np.ndarray) -> np.ndarray:
        """
        :param image:
        :return:
        """
        cv2.rectangle(
            img=image,
            pt1=(66, 66),  # 左上角坐标
            pt2=(99, 99),  # 宽度、高度 99-66=33
            color=(0, 0, 255),  # 线颜色
            thickness=1,  # 线宽度 px
            lineType=1
        )
        cv2.imshow('demo', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return image
