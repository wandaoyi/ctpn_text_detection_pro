#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/28 21:26
# @Author   : WanDaoYi
# @FileName : config.py
# ============================================

import os
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg
cfg = __C

# common options 公共配置文件
__C.COMMON = edict()
# windows 获取文件绝对路径, 方便 windows 在黑窗口 运行项目
__C.COMMON.BASE_PATH = os.path.abspath(os.path.dirname(__file__))
# # 获取当前窗口的路径, 当用 Linux 的时候切用这个，不然会报错。(windows也可以用这个)
# __C.COMMON.BASE_PATH = os.getcwd()

# 相对路径 当前路径
__C.COMMON.RELATIVE_PATH = "./"

__C.COMMON.DATASET_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset")

################################################
# CTPN 配置
################################################

__C.COMMON.FEAT_STRIDE = [16, ]

__C.COMMON.EPS = 1e-14
__C.COMMON.RPN_CLOBBER_POSITIVES = False
__C.COMMON.RPN_NEGATIVE_OVERLAP = 0.3
__C.COMMON.RPN_POSITIVE_OVERLAP = 0.7
__C.COMMON.RPN_FG_FRACTION = 0.5
__C.COMMON.RPN_BATCH_SIZE = 300
__C.COMMON.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
__C.COMMON.RPN_POSITIVE_WEIGHT = -1.0

__C.COMMON.MEAN_LIST = [123.68, 116.78, 103.94]

__C.COMMON.TXT_SUFFIX = ".txt"
__C.COMMON.XML_SUFFIX = ".xml"

#
#
# train options 训练配置文件
__C.TRAIN = edict()
################################################
# CTPN 配置
################################################
__C.TRAIN.GPU_ID = "0"

# 这个路径是制作自己训练数据
__C.TRAIN.ANN_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/annotation")
__C.TRAIN.IMAGE_DATA_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/image_data")
__C.TRAIN.TRAIN_LABEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/train_label")
__C.TRAIN.VAL_LABEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/val_label")
__C.TRAIN.TRAIN_RESIZE_IMAGE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/train_image")
__C.TRAIN.VAL_RESIZE_IMAGE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/val_image")

__C.TRAIN.TRAIN_PERCENT = 0.7
__C.TRAIN.VAL_PERCENT = 0.3

# 这个路径是之间下载现成数据训练
__C.TRAIN.DATA_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/image")
__C.TRAIN.LABEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/label")

__C.TRAIN.LOG_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "logs")
__C.TRAIN.CHECKPOINT_CTPN_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "checkpoints_ctpn")
__C.TRAIN.PRE_TRAINED_MODEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "models/vgg_16.ckpt")

__C.TRAIN.RESTORE = True
__C.TRAIN.LEARNING_RATE = 1e-9

__C.TRAIN.MOVING_AVERAGE_DECAY = 0.997

__C.TRAIN.MAX_STEPS = 60000
__C.TRAIN.DECAY_STEPS = 30000
__C.TRAIN.DECAY_RATE = 0.1
__C.TRAIN.SAVE_CHECKPOINT_STEPS = 200
__C.TRAIN.NUM_READERS = 2

# test options 训练配置文件
__C.TEST = edict()
#####################################################
# CTPN 配置
#####################################################

__C.TEST.INPUT_IMAGE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "dataset/test_image")
__C.TEST.OUTPUT_IMAGE_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "output/ctpn_image")
__C.TEST.OUTPUT_INFO_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "output/ctpn_info")

# test 调用的 GPU 号
__C.TEST.GPU_ID = "0"

# 是否删除原来文件夹内的信息，False 为不删除，True 为删除
__C.TEST.REMOVE_FLAG = True

# 检测模式, 面向过程为 True，面向对象为 False
__C.TEST.DETECT_MODE = True

__C.TEST.MODEL_PATH = os.path.join(__C.COMMON.RELATIVE_PATH, "checkpoints_ctpn")

# 12000,在做nms之前，最多保留的候选box数目
__C.TEST.RPN_PRE_NMS_TOP_N = 12000
# 2000，做完nms之后，最多保留的box的数目
__C.TEST.RPN_POST_NMS_TOP_N = 1000
# nms用参数，阈值是0.7
__C.TEST.RPN_NMS_THRESH = 0.7
# 候选box的最小尺寸，目前是16，高宽均要大于16
__C.TEST.RPN_MIN_SIZE = 8

__C.TEST.MAX_HORIZONTAL_GAP = 50
__C.TEST.PROPOSALS_MIN_SCORE = 0.9
__C.TEST.PROPOSALS_NMS_THRESH = 0.2
__C.TEST.MIN_V_OVERLAPS = 0.7
__C.TEST.MIN_SIZE_SIM = 0.8
__C.TEST.MIN_RATIO = 0.5
__C.TEST.LINE_MIN_SCORE = 0.9
__C.TEST.PROPOSALS_WIDTH = 16
__C.TEST.MIN_NUM_PROPOSALS = 2
