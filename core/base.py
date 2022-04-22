#!/usr/bin/env python
# -*- coding: utf-8 -*-
import urllib.request

import requests

import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    """
    OpenCV读取图片，URL或本地路径读取
    :param path: URL or image_path
    :return: array矩阵: matrix_BGR[x][y] = [B,G,R]
    """
    if path.startswith('http://') or path.startswith('https://'):
        resp = requests.get(url=path).content
        image_ndarray = np.asarray(bytearray(resp), dtype="uint8")
        matrix_BGR = cv2.imdecode(image_ndarray, cv2.IMREAD_COLOR)
    else:
        # 将图像调整为3通道的BGR图像，该值为默认值。cv2.IMREAD_COLOR == 1
        matrix_BGR = cv2.imread(path, cv2.IMREAD_COLOR)
    # cv2.namedWindow('demo')
    # cv2.imshow('demo', matrix_BGR)
    # if cv2.waitKey() == -1:
    #     cv2.destroyWindow('demo')
    #     cv2.destroyAllWindows()
    return matrix_BGR


def write_image(path: str, img_ndarray: np.ndarray, params=None) -> bool:
    """
    保存文件
    :param params:
    :param img_ndarray:
    :param path:
    :return:
    """
    # 上传到URL
    img_encode = cv2.imencode('.png', img_ndarray)[1]
    bytes_encode = np.array(img_encode).tobytes()
    files = {'file': bytes_encode}
    requests.request(method='', url='', files=files)

    # 保存至本地
    flag = cv2.imwrite(filename=path, img=img_ndarray, params=params)
    return flag


def main():
    # 读取图片
    bgr_array = read_image('https://gitee.com/wongqingbin/peaceful/widgets/widget_card.svg')
    # 保存图片副本
    # write_image('11.png', bgr_array)


if __name__ == '__main__':
    main()
