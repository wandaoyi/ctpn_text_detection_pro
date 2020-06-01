#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/28 21:51
# @Author   : WanDaoYi
# @FileName : data_utlis.py
# ============================================

import os
import cv2
import shutil
import numpy as np


def get_data_path(file_path):
    data_path_list = []
    data_name_list = os.listdir(file_path)
    for data_name in data_name_list:
        data_path = os.path.join(file_path, data_name)
        data_path_list.append(data_path)
        pass
    return data_path_list
    pass


def load_ann(file_path):
    bounding_box_list = []
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            line_info = line.strip().split(",")
            x_min, y_min, x_max, y_max = map(int, line_info)
            bounding_box_list.append([x_min, y_min, x_max, y_max, 1])
            pass
        pass
    return bounding_box_list
    pass


def make_file(file_path, remove_flag=False):
    """
        创建文件夹
    :param file_path: 文件夹路径
    :param remove_flag: 是否删除原来已有的文件夹及文件
    :return:
    """
    if remove_flag:
        if os.path.exists(file_path):
            # shutil.rmtree(file_path)
            file_name_list = os.listdir(file_path)
            for file_name in file_name_list:
                path_info = os.path.join(file_path, file_name)
                os.remove(path_info)
                pass
            pass
        else:
            os.mkdir(file_path)
            pass
        pass
    else:
        if not os.path.exists(file_path):
            os.mkdir(file_path)
            pass
        pass
    pass


def resize_image(image, min_size=600, max_size=1200):
    """
        resize image
    :param image: image data
    :param min_size: 目标图像的短边阈值
    :param max_size: 目标图像的长边阈值
    :return:
    """
    img_size = image.shape
    # 图片中的短边
    im_size_min = np.min(img_size[0: 2])
    # 图片中的长边
    im_size_max = np.max(img_size[0: 2])

    im_scale = float(min_size) / float(im_size_min)
    # 如果按短边比例生成的长边大于 长边阈值，则用使用长边来生成比例值
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
        pass
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    # 结果向下取整
    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    # 结果向下取整
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])
    pass

