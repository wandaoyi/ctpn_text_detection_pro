#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/28 22:36
# @Author   : WanDaoYi
# @FileName : anchor_utils.py
# ============================================

import numpy as np
from utils import bbox_utils
from config import cfg

# 是否打印信息
print_param_flag = False


def generate_basic_anchors(sizes, base_size=16):
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    anchors = np.zeros((len(sizes), 4), np.int32)
    index = 0
    for h, w in sizes:
        anchors[index] = scale_anchor(base_anchor, h, w)
        index += 1
    return anchors


def scale_anchor(anchor, h, w):
    x_ctr = (anchor[0] + anchor[2]) * 0.5
    y_ctr = (anchor[1] + anchor[3]) * 0.5
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin
    scaled_anchor[2] = x_ctr + w / 2  # xmax
    scaled_anchor[1] = y_ctr - h / 2  # ymin
    scaled_anchor[3] = y_ctr + h / 2  # ymax
    return scaled_anchor


def generate_anchors():
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16]
    sizes = []
    for h in heights:
        for w in widths:
            sizes.append((h, w))
    return generate_basic_anchors(sizes)


def anchor_target_layer(rpn_cls_score, gt_boxes, image_info, _feat_stride=cfg.COMMON.FEAT_STRIDE):
    """
        Assign anchors to ground-truth targets.
        Produces anchor classification labels and bounding-box regression targets.
    :param rpn_cls_score: (1, H, W, Ax2) bg/fg scores of previous conv layer
    :param gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
    :param image_info: a list of [image_height, image_width, scale_ratios], 如: [[608. 816.   3.]]
    :param _feat_stride: the down_sampling ratio of feature map to the original input image
    :return:
        rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                           that are the regression objectives
        rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
        rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                                beacuse the numbers of bgs and fgs mays significiantly different
    """
    # 生成基本的anchor,一共10个, [xmin, ymin, xmax, ymax]
    # _anchors: [[0, 2, 15, 13], [0, 0, 15,15], [0, -4, 15, 19], [0, -9, 15, 24],
    #            [0, -16, 15, 31], [0, -26, 15, 41], [0, -41, 15, 56],
    #            [0, -62, 15, 77], [0, -91, 15, 106], [0, -134, 15, 149]]
    # _anchors_shape: (10, 4)
    _anchors = generate_anchors()
    # 10 个anchor
    _num_anchors = _anchors.shape[0]

    if print_param_flag:
        print('anchors:')
        print(_anchors)
        print('anchor shapes:')
        print(np.hstack((_anchors[:, 2::4] - _anchors[:, 0::4], _anchors[:, 3::4] - _anchors[:, 1::4])))
        _counts = cfg.COMMON.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0
    # map of shape (..., H, W)
    # height, width = rpn_cls_score.shape[1:3]

    # 图像的高宽及通道数, 如: [[608, 816, 3]] --> [608, 816, 3]
    image_info = image_info[0]
    if print_param_flag:
        print("image_info: ", image_info)
        pass

    assert rpn_cls_score.shape[0] == 1, 'Only single item batches are supported'

    # 在feature-map上定位anchor，并加上delta，得到在实际图像中anchor的真实坐标
    # map of shape (..., H, W), feature-map的高宽, 如: rpn_cls_score_shape: (1, 51, 38, 20)
    height, width = rpn_cls_score.shape[1: 3]

    if print_param_flag:
        print('AnchorTargetLayer: height', height, 'width', width)
        print('')
        print('im_size: ({}, {})'.format(image_info[0], image_info[1]))
        print('scale: {}'.format(image_info[2]))
        print('height, width: ({}, {})'.format(height, width))
        print('rpn: gt_boxes.shape', gt_boxes.shape)
        print('rpn: gt_boxes', gt_boxes)

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # in W H order
    # K is H x W, 生成feature-map和真实image上anchor之间的偏移量
    # np.vstack() 为将两个数组按行放到一起，np.hstack() 为将两个数组按列放到一起
    # 如: a = np.array([[ 8., 8.],[ 0., 0.]]), b = np.array([[ 1., 3.], [ 6., 4.]])
    #   np.vstack((a,b)) --> [[ 8., 8.],[ 0., 0.], [ 1., 3.], [ 6., 4.]]
    #   np.hstack((a,b)) --> [[8. 8. 1. 3.], [0. 0. 6. 4.]]
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    # add number_anchor anchors (1, number_anchor, 4) to
    # cell k shifts (k, 1, 4) to get
    # shift anchors (k, number_anchor, 4)
    # reshape to (k * number_anchor, 4) shifted anchors
    # feature-map的宽乘高的大小, 如: 50 * 37
    # shifts_shape: (1938, 4)
    k = shifts.shape[0]

    # 相当于复制宽高的维度，然后相加;
    # transpose((1, 0, 2)) 将 shape 的维数调换，第 1 维调到 0 下标位置，第 0 维 调到 1 下标位置，
    # 如: shifts_shape: [1, k, 4] --> transpose((1, 0, 2)) --> [k, 1, 4] 的维度。
    # anchor_shape: [1, num_anchors, 4], shifts_shape: [k, 1, 4];
    # [1, num_anchors, 4] + [k, 1, 4] --> [k, num_anchors, 4]
    all_anchors = (_anchors.reshape((1, _num_anchors, 4)) + shifts.reshape((1, k, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((k * _num_anchors, 4))
    total_anchors = int(k * _num_anchors)

    # only keep anchors inside the image
    # 仅保留那些还在图像内部的anchor，超出图像的都删掉, 如 ind_inside_shape: (17328,)
    ind_inside = np.where((all_anchors[:, 0] >= -_allowed_border) &
                          (all_anchors[:, 1] >= -_allowed_border) &
                          (all_anchors[:, 2] < image_info[1] + _allowed_border) &  # width
                          (all_anchors[:, 3] < image_info[0] + _allowed_border)  # height
                          )[0]

    if print_param_flag:
        print('total_anchors', total_anchors)
        print('ind_inside', len(ind_inside))
        pass

    # keep only inside anchors
    # 保留那些在图像内的anchor, anchors_shape: [ind_inside, 4]
    anchors = all_anchors[ind_inside, :]
    if print_param_flag:
        print('anchors.shape', anchors.shape)
        pass

    # 至此，anchor准备好了(所有在图像内的 anchor_boxes)
    # --------------------------------------------------------------

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(ind_inside),), dtype=np.float32)
    # 初始化label，均为-1
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt), shape is A x G
    # 计算anchor和gt-box的overlap，用来给anchor上标签
    # 假设anchors有x个，gt_boxes有y个，返回的是一个（x,y）的数组
    overlaps = bbox_utils.bbox_overlaps(anchors, gt_boxes)

    # 下面的操作类似于 NMS
    # 存放每一个anchor和每一个gtbox之间的overlap, 找到和每一个gtbox，overlap最大的那个anchor
    # argmax_overlaps_shape: [ind_inside, ]; max_overlaps_shape: [ind_inside, ]
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(ind_inside)), argmax_overlaps]

    # 找到每个位置上10个anchor中与 gtbox，overlap最大的那个
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.COMMON.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        # 先给背景上标签，小于0.3 overlap的
        labels[max_overlaps < cfg.COMMON.RPN_NEGATIVE_OVERLAP] = 0
        pass

    # fg label: for each gt, anchor with highest overlap
    # 每个位置上的9个anchor中overlap最大的认为是前景
    labels[gt_argmax_overlaps] = 1
    # fg label: above threshold IOU, overlap大于0.7的认为是前景
    labels[max_overlaps >= cfg.COMMON.RPN_POSITIVE_OVERLAP] = 1

    # 设置背景
    if cfg.COMMON.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.COMMON.RPN_NEGATIVE_OVERLAP] = 0
        pass

    # subsample positive labels if we have too many
    # 对正样本进行采样，如果正样本的数量太多的话
    # 限制正样本的数量不超过128个
    num_fg = int(cfg.COMMON.RPN_FG_FRACTION * cfg.COMMON.RPN_BATCH_SIZE)
    fg_ind = np.where(labels == 1)[0]
    if len(fg_ind) > num_fg:
        # 随机去除掉一些正样本
        disable_ind = np.random.choice(fg_ind, size=(len(fg_ind) - num_fg), replace=False)
        # 变为-1
        labels[disable_ind] = -1
        pass

    # subsample negative labels if we have too many
    # 对负样本进行采样，如果负样本的数量太多的话
    # 正负样本总数是256，限制正样本数目最多128，
    # 如果正样本数量小于128，差的那些就用负样本补上，凑齐256个样本
    num_bg = cfg.COMMON.RPN_BATCH_SIZE - np.sum(labels == 1)
    bg_ind = np.where(labels == 0)[0]
    if len(bg_ind) > num_bg:
        disable_ind = np.random.choice(bg_ind, size=(len(bg_ind) - num_bg), replace=False)
        labels[disable_ind] = -1
        pass

    # 至此， 上好标签，开始计算rpn-box的真值
    # --------------------------------------------------------------

    # bbox_targets = np.zeros((len(ind_inside), 4), dtype=np.float32)
    # 根据anchor和gtbox计算得真值（anchor和gtbox之间的偏差）
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(ind_inside), 4), dtype=np.float32)
    # 内部权重，前景就给1，其他是0
    bbox_inside_weights[labels == 1, :] = np.array(cfg.COMMON.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(ind_inside), 4), dtype=np.float32)
    if cfg.COMMON.RPN_POSITIVE_WEIGHT < 0:  # 暂时使用uniform 权重，也就是正样本是1，负样本是0
        # uniform weighting of examples (given non-uniform sampling)
        # num_examples = np.sum(labels >= 0) + 1
        # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        # negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))
    else:
        assert ((cfg.COMMON.RPN_POSITIVE_WEIGHT > 0) & (cfg.COMMON.RPN_POSITIVE_WEIGHT < 1))

        positive_weights = (cfg.COMMON.RPN_POSITIVE_WEIGHT /
                            (np.sum(labels == 1)) + 1)
        negative_weights = ((1.0 - cfg.COMMON.RPN_POSITIVE_WEIGHT) /
                            (np.sum(labels == 0)) + 1)

    # 外部权重，前景是1，背景是0
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    # 一开始是将超出图像范围的anchor直接丢掉的，现在在加回来
    # 这些anchor的label是-1，也即dontcare
    labels = _unmap(labels, total_anchors, ind_inside, fill=-1)
    # 这些anchor的真值是0，也即没有值
    bbox_targets = _unmap(bbox_targets, total_anchors, ind_inside, fill=0)
    # 内部权重以0填充
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, ind_inside, fill=0)
    # 外部权重以0填充
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, ind_inside, fill=0)

    # labels reshape 一下 label
    labels = labels.reshape((1, height, width, _num_anchors))
    rpn_labels = labels

    # bbox_targets reshape
    bbox_targets = bbox_targets.reshape((1, height, width, _num_anchors * 4))
    rpn_bbox_targets = bbox_targets

    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, _num_anchors * 4))
    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, _num_anchors * 4))
    rpn_bbox_outside_weights = bbox_outside_weights

    if print_param_flag:
        print("anchor target set")
        pass

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, image_info, _feat_stride=cfg.COMMON.FEAT_STRIDE):
    """

    :param rpn_cls_prob_reshape: (1 , H , W , Ax2) outputs of RPN, prob of bg or fg
                            NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
    :param rpn_bbox_pred: (1 , H , W , Ax4), rgs boxes output of RPN
    :param image_info: a list of [image_height, image_width, scale_ratios]
    :param _feat_stride: the downsampling ratio of feature map to the original input image
    :return:
    rpn_rois : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]
        Algorithm:

        for each (H, W) location i
          generate A anchor boxes centered on cell i
          apply predicted bbox deltas at cell i to each of the A anchors
        clip predicted boxes to image
        remove predicted boxes with either height or width < threshold
        sort all (proposal, score) pairs by score from highest to lowest
        take top pre_nms_topN proposals before NMS
        apply NMS with threshold 0.7 to remaining proposals
        take after_nms_topN proposals after NMS
        return the top proposals (-> RoIs top, scores top)
        layer_params = yaml.load(self.param_str_)
    """

    # 生成基本的anchor,一共10个, [xmin, ymin, xmax, ymax]
    # _anchors: [[0, 2, 15, 13], [0, 0, 15,15], [0, -4, 15, 19], [0, -9, 15, 24],
    #            [0, -16, 15, 31], [0, -26, 15, 41], [0, -41, 15, 56],
    #            [0, -62, 15, 77], [0, -91, 15, 106], [0, -134, 15, 149]]
    # _anchors_shape: (10, 4)
    _anchors = generate_anchors()
    # 10 个anchor
    _num_anchors = _anchors.shape[0]

    # 原始图像的高宽、缩放尺度, 如: [[608, 816, 3]] --> [608, 816, 3]
    image_info = image_info[0]

    assert rpn_cls_prob_reshape.shape[0] == 1, 'Only single item batches are supported'

    # 12000,在做nms之前，最多保留的候选box数目
    pre_nms_top_n = cfg.TEST.RPN_PRE_NMS_TOP_N
    # 2000，做完nms之后，最多保留的box的数目
    post_nms_top_n = cfg.TEST.RPN_POST_NMS_TOP_N
    # nms用参数，阈值是0.7
    nms_thresh = cfg.TEST.RPN_NMS_THRESH
    # 候选box的最小尺寸，目前是16，高宽均要大于16
    min_size = cfg.TEST.RPN_MIN_SIZE

    # feature-map的高宽
    height, width = rpn_cls_prob_reshape.shape[1: 3]
    width = width // 10

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    # (1, H, W, A)
    scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, 2])[:, :, :, :, 1],
                        [1, height, width, _num_anchors])
    # 提取到object的分数，non-object的我们不关心

    # 模型输出的pred是相对值，需要进一步处理成真实图像中的坐标
    bbox_deltas = rpn_bbox_pred
    # im_info = bottom[2].data[0, :]

    # 1. Generate proposals from bbox deltas and shifted anchors

    # Enumerate all shifts
    # 同anchor-target-layer-tf这个文件一样，生成anchor的shift，进一步得到整张图像上的所有anchor
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # A = _num_anchors
    k = shifts.shape[0]
    anchors = _anchors.reshape((1, _num_anchors, 4)) + shifts.reshape((1, k, 4)).transpose((1, 0, 2))
    # 这里得到的anchor就是整张图像上的所有anchor
    anchors = anchors.reshape((k * _num_anchors, 4))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    # (HxWxA, 4)
    bbox_deltas = bbox_deltas.reshape((-1, 4))

    # Same story for the scores:
    scores = scores.reshape((-1, 1))

    # Convert anchors into proposals via bbox transformations
    # 做逆变换，得到box在图像上的真实坐标
    proposals = bbox_utils.bbox_transform_inv(anchors, bbox_deltas)

    # 2. clip predicted boxes to image
    # 将所有的proposal修建一下，超出图像范围的将会被修剪掉
    proposals = bbox_utils.clip_boxes(proposals, image_info[: 2])

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    # 移除那些proposal小于一定尺寸的proposal
    keep = _filter_boxes(proposals, min_size)
    # 保留剩下的proposal
    proposals = proposals[keep, :]
    scores = scores[keep]
    bbox_deltas = bbox_deltas[keep, :]

    # # remove irregular boxes, too fat too tall
    # keep = _filter_irregular_boxes(proposals)
    # proposals = proposals[keep, :]
    # scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    # score按得分的高低进行排序
    order = scores.ravel().argsort()[::-1]
    # 保留12000个proposal进去做nms
    if pre_nms_top_n > 0:
        order = order[: pre_nms_top_n]
        pass

    proposals = proposals[order, :]
    scores = scores[order]
    bbox_deltas = bbox_deltas[order, :]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    # 进行nms操作，保留2000个proposal
    keep = bbox_utils.no_cate_nms(np.hstack((proposals, scores)), nms_thresh)
    if post_nms_top_n > 0:
        keep = keep[: post_nms_top_n]
        pass

    proposals = proposals[keep, :]
    scores = scores[keep]
    bbox_deltas = bbox_deltas[keep, :]

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch ind are 0
    blob = np.hstack((scores.astype(np.float32, copy=False), proposals.astype(np.float32, copy=False)))

    return blob, bbox_deltas
    pass


def _unmap(data, count, ind, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[ind] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[ind, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_utils.bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def _filter_irregular_boxes(boxes, min_ratio=0.2, max_ratio=5):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    rs = ws / hs
    keep = np.where((rs <= max_ratio) & (rs >= min_ratio))[0]
    return keep
