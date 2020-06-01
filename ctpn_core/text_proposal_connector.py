#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/30 16:41
# @Author   : WanDaoYi
# @FileName : text_proposal_connector.py
# ============================================

import numpy as np
from utils import bbox_utils
from utils.text_proposal_utils import TextProposalUtils


class TextProposalConnector(object):

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

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            # 求全部小框得分的均值作为文本行的均值
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))

            text_lines[index, 0] = x0
            # 文本行上端 线段 的y坐标的小值
            text_lines[index, 1] = min(lt_y, rt_y)
            text_lines[index, 2] = x1
            # 文本行下端 线段 的y坐标的大值
            text_lines[index, 3] = max(lb_y, rb_y)
            # 文本行得分
            text_lines[index, 4] = score
            pass

        text_lines = bbox_utils.clip_lines_boxes(text_lines, image_size)

        text_recs = np.zeros((len(text_lines), 9), np.float)

        index = 0
        for line in text_lines:
            x_min, y_min, x_max, y_max = line[0], line[1], line[2], line[3]
            text_recs[index, 0] = x_min
            text_recs[index, 1] = y_min
            text_recs[index, 2] = x_max
            text_recs[index, 3] = y_min
            text_recs[index, 4] = x_max
            text_recs[index, 5] = y_max
            text_recs[index, 6] = x_min
            text_recs[index, 7] = y_max
            text_recs[index, 8] = line[4]
            index = index + 1
            pass

        return text_recs
        pass




