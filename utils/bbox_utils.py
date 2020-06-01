#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/28 21:32
# @Author   : WanDaoYi
# @FileName : bbox_utils.py
# ============================================

import numpy as np
from config import cfg


def bbox_overlaps(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_batch = boxes1.shape[0]
    boxes2_batch = boxes2.shape[0]
    overlaps = np.zeros((boxes1_batch, boxes2_batch), dtype=np.float)

    # 计算 面积
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    for i in range(boxes2_batch):
        for j in range(boxes1_batch):
            width_j = min(boxes1[j, 2], boxes2[i, 2]) - max(boxes1[j, 0], boxes2[i, 0]) + 1
            if width_j > 0:
                height_j = min(boxes1[j, 3], boxes2[i, 3]) - max(boxes1[j, 1], boxes2[i, 1]) + 1
                if height_j > 0:
                    union_area = float(boxes1_area[j] + boxes2_area[i] - width_j * height_j)
                    overlaps[j, i] = width_j * height_j / union_area
                    pass
            pass
        pass

    return overlaps
    pass


def no_cate_nms(bboxes, iou_threshold):
    """
        compute text nms
    :param bboxes: (xmin, ymin, xmax, ymax, score)
    :param iou_threshold: iou 阈值
    :return:
    """
    bboxes = np.array(bboxes)

    # 计算 面积
    box_area = (bboxes[..., 2] - bboxes[..., 0]) * (bboxes[..., 3] - bboxes[..., 1])

    bbox_batch = bboxes.shape[0]
    suppressed = np.zeros((bbox_batch,), dtype=np.int)

    scores = bboxes[:, 4]
    order = scores.argsort()[::-1]

    keep = []
    for i in range(bbox_batch):
        n = order[i]
        if suppressed[n] == 1:
            continue
            pass
        keep.append(n)

        xn_1 = bboxes[n, 0]
        yn_1 = bboxes[n, 1]
        xn_2 = bboxes[n, 2]
        yn_2 = bboxes[n, 3]

        for j in range(i + 1, bbox_batch):
            m = order[j]
            if suppressed[m] == 1:
                continue
                pass
            x1 = max(xn_1, bboxes[m, 0])
            y1 = max(yn_1, bboxes[m, 1])
            x2 = min(xn_2, bboxes[m, 2])
            y2 = min(yn_2, bboxes[m, 3])

            w = max(0.0, x2 - x1 + 1)
            h = max(0.0, y2 - y1 + 1)
            inter = w * h
            iou = float(inter / (box_area[n] + box_area[m] - inter))
            if iou >= iou_threshold:
                suppressed[m] = 1
                pass
            pass
        pass

    return keep
    pass


def bbox_transform(ex_rois, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes
    :param gt_rois: n * 4 numpy array, ground-truth boxes
    :return: deltas: n * 4 numpy array, ground-truth boxes
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    assert np.min(ex_widths) > 0.1 and np.min(ex_heights) > 0.1, \
        'Invalid boxes found: {} {}'.format(ex_rois[np.argmin(ex_widths), :], ex_rois[np.argmin(ex_heights), :])

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    # warnings.catch_warnings()
    # warnings.filterwarnings('error')
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets


def bbox_transform_inv(boxes, deltas):
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    # dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    # dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, image_shape):
    """
    Clip boxes to image boundaries.
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], image_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], image_shape[0] - 1), 0)
    # x2 < image_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], image_shape[1] - 1), 0)
    # y2 < image_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], image_shape[0] - 1), 0)
    return boxes


def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)


def clip_lines_boxes(boxes, image_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[:, 0::2] = threshold(boxes[:, 0::2], 0, image_shape[1] - 1)
    boxes[:, 1::2] = threshold(boxes[:, 1::2], 0, image_shape[0] - 1)
    return boxes


def filter_boxes(boxes):
    heights = np.zeros((len(boxes), 1), np.float)
    widths = np.zeros((len(boxes), 1), np.float)
    scores = np.zeros((len(boxes), 1), np.float)
    index = 0
    for box in boxes:
        heights[index] = (abs(box[5] - box[1]) + abs(box[7] - box[3])) / 2.0 + 1
        widths[index] = (abs(box[2] - box[0]) + abs(box[6] - box[4])) / 2.0 + 1
        scores[index] = box[8]
        index += 1

    return np.where((widths / heights > cfg.TEST.MIN_RATIO) & (scores > cfg.TEST.LINE_MIN_SCORE) &
                    (widths > (cfg.TEST.PROPOSALS_WIDTH * cfg.TEST.MIN_NUM_PROPOSALS)))[0]
