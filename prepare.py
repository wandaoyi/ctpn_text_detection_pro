#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/06/01 11:32
# @Author   : WanDaoYi
# @FileName : prepare.py
# ============================================

from datetime import datetime
import os
import cv2
import random
import numpy as np
from utils import data_utlis
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from config import cfg


class Prepare(object):

    def __init__(self):

        self.input_ann_path = cfg.TRAIN.ANN_PATH
        self.input_image_path = cfg.TRAIN.IMAGE_DATA_PATH

        # 输出路径
        self.output_train_label_path = cfg.TRAIN.TRAIN_LABEL_PATH
        self.output_val_label_path = cfg.TRAIN.VAL_LABEL_PATH
        self.output_train_resize_image_path = cfg.TRAIN.TRAIN_RESIZE_IMAGE_PATH
        self.output_val_resize_image_path = cfg.TRAIN.VAL_RESIZE_IMAGE_PATH
        # 如果文件夹不存在，则创建
        data_utlis.make_file(self.output_train_label_path, remove_flag=True)
        data_utlis.make_file(self.output_val_label_path, remove_flag=True)
        data_utlis.make_file(self.output_train_resize_image_path, remove_flag=True)
        data_utlis.make_file(self.output_val_resize_image_path, remove_flag=True)

        # 训练集比例。
        self.train_percent = cfg.TRAIN.TRAIN_PERCENT

        # 后缀名
        self.xml_suffix = cfg.COMMON.XML_SUFFIX
        self.txt_suffix = cfg.COMMON.TXT_SUFFIX

        pass

    def pick_top_left(self, poly):
        idx = np.argsort(poly[:, 0])
        if poly[idx[0], 1] < poly[idx[1], 1]:
            s = idx[0]
        else:
            s = idx[1]

        return poly[(s, (s + 1) % 4, (s + 2) % 4, (s + 3) % 4), :]

    def order_convex(self, p):
        points = Polygon(p).convex_hull
        points = np.array(points.exterior.coords)[:4]
        points = points[::-1]
        points = self.pick_top_left(points)
        points = np.array(points).reshape([4, 2])
        return points

    def shrink_poly(self, poly, r=16):
        # y = kx + b
        x_min = int(np.min(poly[:, 0]))
        x_max = int(np.max(poly[:, 0]))

        k1 = (poly[1][1] - poly[0][1]) / (poly[1][0] - poly[0][0])
        b1 = poly[0][1] - k1 * poly[0][0]

        k2 = (poly[2][1] - poly[3][1]) / (poly[2][0] - poly[3][0])
        b2 = poly[3][1] - k2 * poly[3][0]

        res = []

        start = int((x_min // 16 + 1) * 16)
        end = int((x_max // 16) * 16)

        p = x_min
        res.append([p, int(k1 * p + b1),
                    start - 1, int(k1 * (p + 15) + b1),
                    start - 1, int(k2 * (p + 15) + b2),
                    p, int(k2 * p + b2)])

        for p in range(start, end + 1, r):
            res.append([p, int(k1 * p + b1),
                        (p + 15), int(k1 * (p + 15) + b1),
                        (p + 15), int(k2 * (p + 15) + b2),
                        p, int(k2 * p + b2)])
        return np.array(res, dtype=np.int).reshape([-1, 8])

    def do_prepare(self):
        image_name_list = os.listdir(self.input_image_path)
        print(image_name_list)

        image_list_len = len(image_name_list)
        n_train = int(image_list_len * self.train_percent)
        n_val = image_list_len - n_train

        train_list = random.sample(image_name_list, n_train)

        for image_name in image_name_list:

            image_path = os.path.join(self.input_image_path, image_name)
            image = cv2.imread(image_path)
            image_shape = image.shape
            # re_image, (new_h / image_shape[0], new_w / image_shape[1])
            re_image, (ratio_h, ratio_w) = data_utlis.resize_image(image)
            re_image_shape = re_image.shape

            polys = []

            name_info = os.path.splitext(image_name)[0]
            txt_name = name_info + self.txt_suffix
            xml_name = name_info + self.xml_suffix
            xml_path = os.path.join(self.input_ann_path, xml_name)

            box_info_list = self.convert_ann(xml_path)
            for box_info in box_info_list:
                poly = np.array(box_info).reshape([4, 2])
                # poly[:, 0] = poly[:, 0] / image_shape[1] * re_image_shape[1]
                # poly[:, 1] = poly[:, 1] / image_shape[0] * re_image_shape[0]
                poly[:, 0] = poly[:, 0] * ratio_w
                poly[:, 1] = poly[:, 1] * ratio_h

                poly = self.order_convex(poly)
                polys.append(poly)
                pass

            res_polys = []
            for poly in polys:
                # delete polys with width less than 10 pixel
                if np.linalg.norm(poly[0] - poly[1]) < 10 or np.linalg.norm(poly[3] - poly[0]) < 10:
                    continue

                res = self.shrink_poly(poly)

                res = res.reshape([-1, 4, 2])
                for r in res:
                    x_min = np.min(r[:, 0])
                    y_min = np.min(r[:, 1])
                    x_max = np.max(r[:, 0])
                    y_max = np.max(r[:, 1])

                    res_polys.append([x_min, y_min, x_max, y_max])
                    pass
                pass

            # 设置保存路径
            if image_name in train_list:
                output_image_path = os.path.join(self.output_train_resize_image_path, image_name)
                output_label_path = os.path.join(self.output_train_label_path, txt_name)
                pass
            else:
                output_image_path = os.path.join(self.output_val_resize_image_path, image_name)
                output_label_path = os.path.join(self.output_val_label_path, txt_name)
                pass

            cv2.imwrite(os.path.join(output_image_path), re_image)
            with open(os.path.join(output_label_path), "w") as f:
                for p in res_polys:
                    line = ",".join(str(p[i]) for i in range(4)) + "\n"
                    f.writelines(line)
            pass
        pass

    def convert_ann(self, xml_path):

        box_info = []
        print("xml_path: {}".format(xml_path))
        # 打开 xml 文件
        xml_file = open(xml_path)
        # 将 xml 文件 转为树状结构
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.iter("object"):
            # cls = obj.find("name").text
            xml_box = obj.find("bndbox")
            xmin = int(xml_box.find('xmin').text)
            ymin = int(xml_box.find('ymin').text)
            xmax = int(xml_box.find('xmax').text)
            ymax = int(xml_box.find('ymax').text)

            # 以左上角为基础，顺时针旋转找角点。(左上, 右上, 右下, 左下)
            box_info.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
            pass

        return box_info
        pass


if __name__ == "__main__":

    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = Prepare()
    demo.do_prepare()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
    pass
