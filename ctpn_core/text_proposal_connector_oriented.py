#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/30 16:41
# @Author   : WanDaoYi
# @FileName : text_proposal_connector_oriented.py
# ============================================

import numpy as np
from utils.text_proposal_utils import TextProposalUtils


class TextProposalConnectorOriented(object):

    def __init__(self):
        self.text_proposal_utils = TextProposalUtils()
        pass

    def get_text_lines(self, text_proposals, scores, image_size):

        # 首先还是建图，获取到文本行由哪几个小框构成
        groups = self.text_proposal_utils.build_graph(text_proposals, scores, image_size)
        tp_groups = self.text_proposal_utils.sub_graphs_connected(groups)
        text_lines = np.zeros((len(tp_groups), 5), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            # 每个文本行的全部小框
            text_line_boxes = text_proposals[list(tp_indices)]
            # 求每一个小框的中心x，y坐标
            x = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2
            y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2

            # 多项式拟合，根据之前求的中心店拟合一条直线（最小二乘）
            z1 = np.polyfit(x, y, 1)

            # 文本行x坐标最小值
            x0 = np.min(text_line_boxes[:, 0])
            # 文本行x坐标最大值
            x1 = np.max(text_line_boxes[:, 2])

            # 小框宽度的一半
            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5

            # 以全部小框的左上角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lt_y, rt_y = self.text_proposal_utils.fit_y(text_line_boxes[:, 0],
                                                        text_line_boxes[:, 1],
                                                        x0 + offset,
                                                        x1 - offset)

            # 以全部小框的左下角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lb_y, rb_y = self.text_proposal_utils.fit_y(text_line_boxes[:, 0],
                                                        text_line_boxes[:, 3],
                                                        x0 + offset,
                                                        x1 - offset)

            # 求全部小框得分的均值作为文本行的均值
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)
            # 文本行上端 线段 的y坐标的小值
            text_lines[index, 2] = x1
            # 文本行下端 线段 的y坐标的大值
            text_lines[index, 3] = max(lb_y, rb_y)
            # 文本行得分
            text_lines[index, 4] = score
            # 根据中心点拟合的直线的k，b
            text_lines[index, 5] = z1[0]
            text_lines[index, 6] = z1[1]
            # 小框平均高度
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))
            text_lines[index, 7] = height + 2.5

        text_recs = np.zeros((len(text_lines), 9), np.float)

        index = 0
        for line in text_lines:
            # 根据高度和文本行中心线，求取文本行上下两条线的b值
            b1 = line[6] - line[7] / 2
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            # 左上
            y1 = line[5] * line[0] + b1
            x2 = line[2]
            # 右上
            y2 = line[5] * line[2] + b1
            x3 = line[0]
            # 左下
            y3 = line[5] * line[0] + b2
            x4 = line[2]
            # 右下
            y4 = line[5] * line[2] + b2

            dis_x = x2 - x1
            dis_y = y2 - y1
            # 文本行宽度
            width = np.sqrt(dis_x * dis_x + dis_y * dis_y)

            # 文本行高度
            f_tmp0 = y3 - y1
            f_tmp1 = f_tmp0 * dis_y / width

            # 做补偿
            x = np.fabs(f_tmp1 * dis_x / width)
            y = np.fabs(f_tmp1 * dis_y / width)

            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x4
            text_recs[index, 5] = y4
            text_recs[index, 6] = x3
            text_recs[index, 7] = y3
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs
        pass
