import pandas as pd
import torch
import numpy as np
from typing import List

def label_map(label_map_path):
    label_map = {}

    with open(label_map_path, 'r', encoding='utf-8') as t:
        for i, line in enumerate(t.readlines()):

            line = line.rstrip('\n')
            label_map[i] = line

    return label_map

class MetricTracker(object):
    def __init__(self, label_map: List[str], writer=None):
        self.label_map = label_map
        self.label_name = list(self.label_map.keys())
        self.switch_kv_label_map = {v: k for k, v in self.label_map.items()}
        self.writer = writer
        self.confusion_metric = pd.DataFrame(0, index=self.label_name, columns=self.label_name)
        self.reset()

    def reset(self):
        for col in self.confusion_metric.columns:
            self.confusion_metric[col].values[:] = 0

    def update(self, target, preds):
        pred = torch.argmax(preds, dim=1)
        target = target.cpu().data.numpy()
        pred = pred.cpu().data.numpy()
        for i in range(len(target)):
            self.confusion_metric.loc[self.label_map[target[i]],
                                  self.label_map[pred[i]]] += 1

    def result(self):
        return self.confusion_metric

    def accuracy(self):
        ACC_PER_CATEGORY = {}
        mAP, mAR, TOTAL_F1_SCORE, TOTAL_ACC = [], [], [], []

        for l in self.switch_kv_label_map:
            ok_cnt = self.confusion_metric.loc[l, l]
            c_values = self.confusion_metric.loc[:, l].values
            r_values = self.confusion_metric.loc[l, :].values
            diff_values = self.confusion_metric.loc[self.confusion_metric.columns != l,
                                                    self.confusion_metric.columns != l].values

            # 0으로 초기화 했던 자리가 업데이터가 안될경우 예외처리.
            if ok_cnt == 0 or np.sum(c_values) == 0 or np.sum(r_values) == 0:
                continue

            AP = ok_cnt / np.sum(c_values)
            AR = ok_cnt / np.sum(r_values)
            F1_SCORE = 2 * (AP * AR) / (AP + AR)
            ACC = (ok_cnt + np.sum(diff_values)) / np.sum(np.array(self.confusion_metric))
            ACC_PER_CATEGORY[l] = {'AP': AP,
                                   'AR': AR,
                                   'F1_SCORE': F1_SCORE,
                                   'ACC': ACC}
            mAP.append(AP), mAR.append(AR), TOTAL_F1_SCORE.append(F1_SCORE), TOTAL_ACC.append(ACC)
        true = np.sum(np.diag(np.array(self.confusion_metric)))
        total_cnt = np.sum(np.array(self.confusion_metric))
        ACC = true / total_cnt
        return ACC_PER_CATEGORY, np.mean(mAP), np.mean(mAR), np.mean(TOTAL_F1_SCORE), ACC