#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/28 22:26
# @Author   : WanDaoYi
# @FileName : ctpn_model.py
# ============================================

import tensorflow as tf
from tensorflow.contrib import slim

from model_net import vgg_net
from utils.anchor_utils import anchor_target_layer as anchor_target_layer_py
from config import cfg


def mean_image_subtraction(images, means=cfg.COMMON.MEAN_LIST):
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def make_var(name, shape, initializer=None):
    return tf.get_variable(name, shape, initializer=initializer)


def Bilstm(net, input_channel, hidden_unit_num, output_channel, scope_name):
    """
        Bi-directional LSTM (BLSTM)
    :param net: backbone 输出网络
    :param input_channel: 输入通道
    :param hidden_unit_num: 隐藏层通道数
    :param output_channel: 输出通道
    :param scope_name: name
    :return:
    """
    # width--->time step
    with tf.variable_scope(scope_name):
        shape = tf.shape(net)
        n, h, w, c = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [n * h, w, c])
        net.set_shape([None, None, input_channel])

        # 长短时记忆
        lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)

        # The sequential windows in each row are recurrently connected by a Bi-directional LSTM (BLSTM)
        lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, net, dtype=tf.float32)
        lstm_out = tf.concat(lstm_out, axis=-1)

        lstm_out = tf.reshape(lstm_out, [n * h * w, 2 * hidden_unit_num])

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [2 * hidden_unit_num, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        outputs = tf.matmul(lstm_out, weights) + biases

        outputs = tf.reshape(outputs, [n, h, w, output_channel])
        return outputs


def lstm_fc(net, input_channel, output_channel, scope_name):
    """
        从长短时记忆输出到全连接输出
    :param net: 长短时记忆输出网络
    :param input_channel: 输入通道
    :param output_channel: 输出通道
    :param scope_name: name
    :return:
    """
    with tf.variable_scope(scope_name):
        shape = tf.shape(net)
        n, h, w, c = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [n * h * w, c])

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [input_channel, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        output = tf.matmul(net, weights) + biases
        output = tf.reshape(output, [n, h, w, output_channel])
    return output


def model(image):

    # 将图像均值化
    image = mean_image_subtraction(image)

    with slim.arg_scope(vgg_net.vgg_arg_scope()):
        # 将图像经过 vgg16 backbone network
        conv5_3 = vgg_net.vgg_16(image)
        pass

    rpn_conv = slim.conv2d(conv5_3, 512, 3)

    #  Bi-directional LSTM (BLSTM)
    lstm_output = Bilstm(rpn_conv, 512, 128, 512, scope_name='BiLSTM')

    # 将长短时记忆输出结果 进行全连接 输出预测
    bbox_pred = lstm_fc(lstm_output, 512, 10 * 4, scope_name="bbox_pred")
    cls_pred = lstm_fc(lstm_output, 512, 10 * 2, scope_name="cls_pred")

    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])

    cls_pred_reshape_shape = tf.shape(cls_pred_reshape)
    cls_prob = tf.reshape(tf.nn.softmax(tf.reshape(cls_pred_reshape, [-1, cls_pred_reshape_shape[3]])),
                          [-1, cls_pred_reshape_shape[1], cls_pred_reshape_shape[2], cls_pred_reshape_shape[3]],
                          name="cls_prob")

    return bbox_pred, cls_pred, cls_prob


def anchor_target_layer(cls_pred, bbox, im_info, scope_name):
    with tf.variable_scope(scope_name):
        # 'rpn_cls_score', 'gt_boxes', 'im_info'
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            tf.py_func(anchor_target_layer_py,
                       [cls_pred, bbox, im_info, [16, ]],
                       [tf.float32, tf.float32, tf.float32, tf.float32])

        rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name='rpn_labels')
        rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name='rpn_bbox_targets')
        rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights, name='rpn_bbox_inside_weights')
        rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights, name='rpn_bbox_outside_weights')

        return [rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights]


def smooth_l1_dist(deltas, sigma2=9.0, name='smooth_l1_dist'):
    with tf.name_scope(name=name):
        deltas_abs = tf.abs(deltas)
        smooth_l1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
        return tf.square(deltas) * 0.5 * sigma2 * smooth_l1_sign + \
               (deltas_abs - 0.5 / sigma2) * tf.abs(smooth_l1_sign - 1)


def loss(bbox_pred, cls_pred, bbox, im_info):

    rpn_data = anchor_target_layer(cls_pred, bbox, im_info, "anchor_target_layer")

    # classification loss
    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])
    rpn_cls_score = tf.reshape(cls_pred_reshape, [-1, 2])
    rpn_label = tf.reshape(rpn_data[0], [-1])

    # ignore_label(-1)
    fg_keep = tf.equal(rpn_label, 1)
    rpn_keep = tf.where(tf.not_equal(rpn_label, -1))
    rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep)
    rpn_label = tf.gather(rpn_label, rpn_keep)
    rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)

    # box loss
    rpn_bbox_pred = bbox_pred
    rpn_bbox_targets = rpn_data[1]
    rpn_bbox_inside_weights = rpn_data[2]
    rpn_bbox_outside_weights = rpn_data[3]

    # shape (N, 4)
    rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep)
    rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep)
    rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep)
    rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep)

    rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * smooth_l1_dist(
        rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), reduction_indices=[1])

    rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
    rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)

    model_loss = rpn_cross_entropy + rpn_loss_box

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(regularization_losses) + model_loss

    tf.summary.scalar('model_loss', model_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('rpn_cross_entropy', rpn_cross_entropy)
    tf.summary.scalar('rpn_loss_box', rpn_loss_box)

    return total_loss, model_loss, rpn_cross_entropy, rpn_loss_box
