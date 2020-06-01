#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/28 22:28
# @Author   : WanDaoYi
# @FileName : ctpn_train.py
# ============================================

from datetime import datetime
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from ctpn_core import dataset, ctpn_model
from config import cfg

# 使用 0 号 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAIN.GPU_ID


class CTPNTrain(object):

    def __init__(self):

        self.data_path = cfg.TRAIN.DATA_PATH
        self.label_path = cfg.TRAIN.LABEL_PATH

        # 日志保存路径
        self.log_path = self.log_file_path(cfg.TRAIN.LOG_PATH)
        self.checkpoint_path = cfg.TRAIN.CHECKPOINT_CTPN_PATH

        # 如果文件夹不存在则创建
        self.make_dir(self.log_path)
        self.make_dir(self.checkpoint_path)

        self.learning_rate = cfg.TRAIN.LEARNING_RATE
        self.pre_trained_model_path = cfg.TRAIN.PRE_TRAINED_MODEL_PATH
        self.restore = cfg.TRAIN.RESTORE

        self.moving_average_decay = cfg.TRAIN.MOVING_AVERAGE_DECAY

        self.max_steps = cfg.TRAIN.MAX_STEPS
        self.decay_steps = cfg.TRAIN.DECAY_STEPS
        self.decay_rate = cfg.TRAIN.DECAY_RATE
        self.save_checkpoint_steps = cfg.TRAIN.SAVE_CHECKPOINT_STEPS
        self.num_readers = cfg.TRAIN.NUM_READERS
        pass

    # 设置 日志文件夹
    def log_file_path(self, log_dir):
        log_start_time = datetime.now()
        log_file_name = "{:%Y%m%dT%H%M}".format(log_start_time)
        log_path = os.path.join(log_dir, log_file_name)

        return log_path
        pass

    def make_dir(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            pass
        pass

    def do_train(self):

        # 构建 画图
        # input_image 为 图像的 3 个通道中每个像素点的详细信息。
        # input_image_shape: [batch_size, height, width, channel], 例如: [1, 608, 816, 3]
        # 详情参考 dataset.py 中的 generator() 方法
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        # ground truth box: [[xmin, ymin, xmax, ymax, label], ..., [xmin, ymin, xmax, ymax, label]]
        # 例如: [[224, 239, 239, 347, 1], ..., [256, 239, 271, 347, 1]], 1 表示前景
        # 例如: input_bbox_shape: [23, 5]
        input_bbox = tf.placeholder(tf.float32, shape=[None, 5], name='input_bbox')
        # 对 input_image 的 描述，比方说，[[height_1, width_1, channel], ..., [height_n, width_n, channel]]
        # input_image_info_shape: [batch_size, 3], 例如: [1, 3]
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.Variable(self.learning_rate, trainable=False)
        tf.summary.scalar('learning_rate', learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate)

        gpu_id = int(cfg.TRAIN.GPU_ID)
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                bbox_pred, cls_pred, cls_prob = ctpn_model.model(input_image)
                total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = ctpn_model.loss(bbox_pred,
                                                                                          cls_pred,
                                                                                          input_bbox,
                                                                                          input_im_info)

                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        summary_op = tf.summary.merge_all()
        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, global_step)

        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
            train_op = tf.no_op(name='train_op')

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        summary_writer = tf.summary.FileWriter(self.log_path, tf.get_default_graph())

        init = tf.global_variables_initializer()

        if self.pre_trained_model_path is not None:
            variable_restore_op = slim.assign_from_checkpoint_fn(self.pre_trained_model_path,
                                                                 slim.get_trainable_variables(),
                                                                 ignore_missing_vars=True)
            pass
        else:
            variable_restore_op = None
            pass

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.allow_soft_placement = True

        with tf.Session(config=config) as sess:
            if self.restore:
                ckpt = tf.train.latest_checkpoint(self.checkpoint_path)
                print("=========================", ckpt)

                file_name = os.path.split(ckpt)[-1]
                restore_step = int(file_name.split('.')[0].split('_')[-1])
                print("continue training from previous checkpoint {}".format(restore_step))
                saver.restore(sess, ckpt)
            else:
                sess.run(init)
                restore_step = 0
                if self.pre_trained_model_path is not None:
                    variable_restore_op(sess)

            data_generator = dataset.get_batch(self.data_path, self.label_path, num_workers=self.num_readers)
            start = time.time()
            print("start run...")
            for step in range(restore_step, self.max_steps):
                data = next(data_generator)
                # print("data_shape: {}".format(np.array(data[0]).shape))
                ml, tl, _, summary_str = sess.run([model_loss, total_loss, train_op, summary_op],
                                                  feed_dict={input_image: data[0],
                                                             input_bbox: data[1],
                                                             input_im_info: data[2]})

                summary_writer.add_summary(summary_str, global_step=step)

                if step != 0 and step % self.decay_steps == 0:
                    sess.run(tf.assign(learning_rate, learning_rate.eval() * self.decay_rate))

                if step % 10 == 0:
                    avg_time_per_step = (time.time() - start) / 10
                    start = time.time()
                    print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, LR: {:.6f}'.format(
                        step, ml, tl, avg_time_per_step, learning_rate.eval()))

                if (step + 1) % self.save_checkpoint_steps == 0:
                    filename = ('ctpn_{:d}'.format(step + 1) + '.ckpt')
                    filename = os.path.join(self.checkpoint_path, filename)
                    saver.save(sess, filename)
                    print('Write model to: {:s}'.format(filename))
        pass


if __name__ == "__main__":
    # 代码开始时间
    start_time = datetime.now()
    print("开始时间: {}".format(start_time))

    demo = CTPNTrain()
    demo.do_train()

    # 代码结束时间
    end_time = datetime.now()
    print("结束时间: {}, 训练模型耗时: {}".format(end_time, end_time - start_time))
    pass





