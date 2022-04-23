#!/usr/bin/env python
# -*- coding: utf-8 -*-
from core.base import PyImage


def compare(b: str, c: str):
    image = PyImage()

    # 读取数据
    benchmark = image.read(b)
    comparison = image.read(c)

    # 判断大小两张图片大小是否一致
    if benchmark.shape != comparison.shape:
        print("图片大小不一致")
        return

    # 灰度处理
    gray_b = image.to_gray(benchmark)
    gray_c = image.to_gray(comparison)

    # 二值处理
    binary_b = image.to_binary(gray_b)
    binary_c = image.to_binary(gray_c)

    # 异或结果
    results = image.xor(binary_b, binary_c)

    # 滤波
    results = image.denoising(results)

    # 画框
    image.draw_box(benchmark)


if __name__ == '__main__':
    compare('/Users/frieza/Desktop/3.png', '/Users/frieza/Desktop/4.png')
