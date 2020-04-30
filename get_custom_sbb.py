#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'get custom data ground truth'
'''
:input: img gt csv
:return: 
'''

import pandas as pd
import os
import sys
import math
import json
import numpy as np
import torch
import csv


def get_custom_labels_imgpathsfile():
    def xyxy2xywh(x):
        x = list(map(float, x))
        # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[0] = (x[0] + x[2]) / 2
        y[1] = (x[1] + x[3]) / 2
        y[2] = x[2] - x[0]
        y[3] = x[3] - x[1]
        # Normalize coordinates 0 - 1
        y[0] = round(y[0] / w,4)# width #大图片的宽
        y[2] = round(y[2] / w,4)
        y[1] = round(y[1] / h,4) # height
        y[3] = round(y[3] / h,4)
        return y

    label_dir = 'training_gt_1080p_v1106'
    csv_path = 'nba_train_1023.csv'

    w, h = 1960,1080

    df = pd.read_csv(csv_path, encoding='utf-8', sep=',', header=None, names=['img','xy','gt'])
    df['img_name'] = df['img'].apply(lambda x: x.split("/")[-1])
    df['labels'] = df['gt'].apply(lambda x: str(x).split("_"))

    for index,lines in enumerate(df['xy']):
        vertices = []
        labels = []
        points = list(map(int, lines.strip().split("_")))
        if len(points) % 4 != 0:
            print(lines)
        print(index)
        assert len(points) >= 8 and len(points) % 4 == 0
        x1, y1, x2, y2 = points[:4]
        length = len(points)
        for i in range(4, length, 4):
            vertices.append(
                [points[i + 0], points[i + 1], points[i + 2], points[i + 1],
                 points[i + 2], points[i + 3],points[i + 0],points[i + 3]])
        for j in df['labels'][index]:
            labels.append(j)
        df_gt = pd.DataFrame(list(zip(vertices, labels)))

        df_gt['xy'] = df_gt[0]  #df_gt[0]#.apply(lambda x: ','.join(list(map(str,x))).replace('[','').replace(']',''))
        df_gt['gt'] = df_gt[1]
        df_gt = pd.concat( [ pd.DataFrame(df_gt['xy'].values.tolist(), columns=['xy_' + str(x) for x in range(len(df_gt.loc[0,'xy']))]), df_gt['gt'] ], axis=1)
        print(df_gt)
        w_path = label_dir + df['img_name'][index] + '.txt'
        df_gt.to_csv(w_path,index=0,header=0)

if __name__ == '__main__':
    get_custom_labels_imgpathsfile()
