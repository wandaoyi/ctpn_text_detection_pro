#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/30 17:52
# @Author   : WanDaoYi
# @FileName : ctpn_test.py
# ============================================

from datetime import datetime
import os
import cv2
import numpy as np
import tensorflow as tf
from utils import data_utlis, bbox_utils
from ctpn_core import ctpn_model
from utils.anchor_utils import proposal_layer
from ctpn_core.text_proposal_connector import TextProposalConnector
from ctpn_core.text_proposal_connector_oriented import TextProposalConnectorOriented
from config import cfg

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.TEST.GPU_ID


class CTPNTest(object):

    def __init__(self):

        self.test_data_path = cfg.TEST.INPUT_IMAGE_PATH
        self.output_image_path = cfg.TEST.OUTPUT_IMAGE_PATH
        self.output_info_path = cfg.TEST.OUTPUT_INFO_PATH
        self.model_path = cfg.TEST.MODEL_PATH

        self.remove_flag = cfg.TEST.REMOVE_FLAG

        self.detect_mode = cfg.TEST.DETECT_MODE

        if self.detect_mode:
            self.text_detector = TextProposalConnector()
            pass
        else:
            self.text_detector = TextProposalConnectorOriented()
            pass
        pass

    def detect(self, text_proposals, scores, size):
        # 删除得分较低的proposal
        keep_ind = np.where(scores > cfg.TEST.PROPOSALS_MIN_SCORE)[0]
        text_proposals, scores = text_proposals[keep_ind], scores[keep_ind]

        # 按得分排序
        sorted_indices = np.argsort(scores.ravel())[::-1]
        text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]

        # 对proposal做nms
        keep_ind = bbox_utils.no_cate_nms(np.hstack((text_proposals, scores)), cfg.TEST.PROPOSALS_NMS_THRESH)
        text_proposals, scores = text_proposals[keep_ind], scores[keep_ind]

        # 获取检测结果
        text_recs = self.text_detector.get_text_lines(text_proposals, scores, size)
        keep_ind = bbox_utils.filter_boxes(text_recs)

        return text_recs[keep_ind]
        pass

    def do_test(self):
        print("start....")
        # 创建输出文件夹
        data_utlis.make_file(self.output_image_path, self.remove_flag)
        data_utlis.make_file(self.output_info_path, self.remove_flag)

        with tf.get_default_graph().as_default():
            input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
            input_image_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

            bbox_pred, cls_pred, cls_prob = ctpn_model.model(input_image)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            gpu_options = tf.GPUOptions(allow_growth=True)
            config_info = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
            with tf.Session(config=config_info) as sess:
                ckpt_state = tf.train.get_checkpoint_state(self.model_path)
                model_path = os.path.join(self.model_path, os.path.basename(ckpt_state.model_checkpoint_path))
                print('Restore from {}'.format(model_path))
                saver.restore(sess, model_path)

                image_path_list = data_utlis.get_data_path(self.test_data_path)
                for image_path in image_path_list:
                    image = cv2.imread(image_path)
                    resize_image, (ratio_h, ratio_w) = data_utlis.resize_image(image)

                    rh, rw, c = resize_image.shape
                    resize_image_info = np.array([rh, rw, c]).reshape([1, 3])

                    bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                           feed_dict={input_image: [resize_image],
                                                                      input_image_info: resize_image_info})

                    text_seg, _ = proposal_layer(cls_prob_val, bbox_pred_val, resize_image_info)

                    scores = text_seg[:, 0]
                    text_seg = text_seg[:, 1: 5]

                    boxes = self.detect(text_seg, scores[:, np.newaxis], resize_image.shape[: 2])

                    boxes = np.array(boxes, dtype=np.int)

                    for i, box in enumerate(boxes):
                        cv2.polylines(resize_image, [box[: 8].astype(np.int32).reshape((-1, 1, 2))], True,
                                      color=(0, 255, 0), thickness=2)
                        pass

                    reduction_image = cv2.resize(resize_image, None, None, fx=1.0 / ratio_h, fy=1.0 / ratio_w,
                                                 interpolation=cv2.INTER_LINEAR)

                    output_image_path = os.path.join(self.output_image_path, os.path.basename(image_path))
                    cv2.imwrite(output_image_path, reduction_image[:, :, ::-1])

                    image_name = os.path.split(image_path)[-1]
                    name_info = os.path.splitext(image_name)[0]
                    output_txt_path = os.path.join(self.output_info_path, name_info + ".txt")
                    with open(output_txt_path, "w") as file:
                        for i, box in enumerate(boxes):
                            line = ",".join(str(box[k]) for k in range(8))
                            line += "," + str(scores[i]) + "\r\n"
                            file.writelines(line)
                            pass
                    pass
        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = CTPNTest()
    demo.do_test()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
    pass
