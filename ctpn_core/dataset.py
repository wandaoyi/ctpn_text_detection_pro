#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/28 22:05
# @Author   : WanDaoYi
# @FileName : dataset.py
# ============================================

import os
import cv2
import time
import platform
import numpy as np
import matplotlib.pyplot as plt
from utils import data_utlis
from utils.generator_queue import GeneratorQueue


def generator(data_path, ann_path, vis_flag=False):
    """
        数据生成器
    :param data_path: image data file path
    :param ann_path: label data file path
    :param vis_flag: 是否可视化，True 为可视化展示
    :return:
    """
    image_path_list = np.array(data_utlis.get_data_path(data_path))
    image_number = image_path_list.shape[0]
    print("{} training images in {}".format(image_number, data_path))
    index = np.arange(image_number)
    txt_suffix = ".txt"
    while True:
        np.random.shuffle(index)
        for i in index:
            try:
                image_path = image_path_list[i]
                image = cv2.imread(image_path)
                h, w, c = image.shape
                image_info = np.array([h, w, c]).reshape([1, 3])

                # os.path.split("./dataset/image/100_icdar13.png")
                #  --> ('./dataset/image', '100_icdar13.png')
                image_name = os.path.split(image_path)[-1]
                name_info = image_name.split(".")[0]
                label_path = os.path.join(ann_path, name_info + txt_suffix)
                if not os.path.exists(label_path):
                    print("ground truth for image {} not exist!".format(image_path))
                    continue
                    pass
                bounding_box_list = data_utlis.load_ann(label_path)
                if len(bounding_box_list) == 0:
                    print("ground truth for image {} empty!".format(image_path))
                    continue
                    pass
                if vis_flag:
                    for bounding_box in bounding_box_list:
                        cv2.rectangle(image, (bounding_box[0], bounding_box[1]),
                                      (bounding_box[2], bounding_box[3]),
                                      color=(0, 0, 255), thickness=1)
                        pass
                    fig, axs = plt.subplots(1, 1, figsize=(30, 30))
                    axs.imshow(image[:, :, ::-1])
                    axs.set_xticks([])
                    axs.set_yticks([])
                    plt.tight_layout()
                    plt.show()
                    plt.close()
                    pass

                yield [image], bounding_box_list, image_info
                pass
            except Exception as e:
                print(e)
                continue
        pass
    pass


def get_batch(data_path, ann_path, num_workers, **kwargs):
    queue_num = None
    try:
        if platform.system() == "Windows":
            # 在 windows 下存在进程启动问题，windows 上不支持多核操作，
            # windows 无法编辑包含生成器的对象
            # 使用多线程还有点故障, 所以改为使用 主线程。
            # 需要将 use_multiprocessing=False, workers=1 或 use_multiprocessing=True, workers=0
            # 参考: https://github.com/keras-team/keras/pull/8662  QueueUtils is unusable on Windows.
            # 参考: https://github.com/matterport/Mask_RCNN/issues/93
            # 这个 use_multiprocessing=False, workers=1, max_queue_size=1 固定设置，只算是暂时解决问题而已
            queue_num = GeneratorQueue(generator(data_path, ann_path, **kwargs), use_multiprocessing=False)
            queue_num.start(workers=1, max_queue_size=4)
            pass
        else:
            # 在Linux 下可以正常使用多线程
            queue_num = GeneratorQueue(generator(data_path, ann_path, **kwargs), use_multiprocessing=True)
            queue_num.start(workers=num_workers, max_queue_size=24)
            pass

        generator_output = None

        while True:
            while queue_num.is_running():
                if not queue_num.queue.empty():
                    generator_output = queue_num.queue.get()
                    break
                    pass
                else:
                    time.sleep(0.01)
                    pass

            yield generator_output
            generator_output = None
            pass
        pass
    finally:
        if queue_num is not None:
            queue_num.stop()
        pass
    pass


if __name__ == "__main__":

    image_data_path = "../dataset/image"
    ann_info_path = "../dataset/label"
    # window 下 num_workers 改为 0
    batch_data = get_batch(image_data_path, ann_info_path, num_workers=2, vis_flag=True)
    while True:
        images, bbox, image_infos = next(batch_data)
        # print(image, bbox, image_info)
        print("done! ")
        pass
    pass
