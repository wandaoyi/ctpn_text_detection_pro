#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# ============================================
# @Time     : 2020/05/30 13:06
# @Author   : WanDaoYi
# @FileName : text_proposal_utils.py
# ============================================

import numpy as np
from config import cfg


class TextProposalUtils(object):

    def __init__(self):
        self.text_proposals = None
        self.scores = None
        self.im_size = None
        self.heights = None
        self.boxes_table = None
        pass

    def sub_graphs_connected(self, graph):
        sub_graphs = []
        for index in range(graph.shape[0]):
            if not graph[:, index].any() and graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while graph[v, :].any():
                    v = np.where(graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
                    pass
                pass
            pass
        return sub_graphs
        pass

    def build_graph(self, text_proposals, scores, image_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = image_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
            pass

        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
                pass

            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                graph[index, succession_index] = True
                pass
            pass

        return graph
        pass

    def get_successions(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + cfg.TEST.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - cfg.TEST.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        """
            试过将 overlaps_v(), size_similarity() 抽出来，发现检测不理想。
        """
        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= cfg.TEST.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= cfg.TEST.MIN_SIZE_SIM

    def fit_y(self, x, y, x1, x2):
        # if x only include one point, the function will get line y=Y[0]
        if np.sum(x == x[0]) == len(x):
            return y[0], y[0]
        p = np.poly1d(np.polyfit(x, y, 1))
        return p(x1), p(x2)
