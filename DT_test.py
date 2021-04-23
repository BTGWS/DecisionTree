#!/usr/bin/env python
# coding:utf-8
# @TIME         : 2021/4/20 3:17 下午
# @Author       : BTG
# @Project      : pythonProject
# @File Name    : DT_test.py
"""
# Code is far away from bugs with the god animal protecting.
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃        ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import time

import pandas as pd
import sys

from alive_progress import alive_bar

from ReadFiles import read_file
from Functions import data_split, floatDataSplit
from Decision_Tree import DecisionTree_ID3
import pprint as pt

"""    
    print(dataset.columns.values)   # 通过columns字段获取，返回一个numpy型的array
    print(list(dataset))            # 直接使用 list 关键字，返回一个list
    print(dataset.columns.tolist()) # df.columns 返回Index，可以通过 tolist(), 或者 list（array） 转换为list
"""
start_head = '../My_Datasets/watermelon/'
sfile_name = ['watermelon2.csv', 'watermelon2_1.csv', 'watermelon3.csv', 'watermelon3_1.csv']

if __name__ == '__main__':
    dataset_wm2 = read_file(start_head + sfile_name[0])
    dataset_wm3 = read_file(start_head + sfile_name[2])
    dataset_columnsNames_wm2 = dataset_wm2.columns.tolist()[1:]  # ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    dataset2 = dataset_wm2[dataset_columnsNames_wm2]

    # ==================================================================
    # 将数据中的连续值离散化后重新加入
    # 连续值处理方法
    temp_data_x1 = dataset_wm3[['密度']]
    temp_data_x2 = dataset_wm3[['含糖率']]
    dataset_columnsNames_wm3 = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    dataset3 = dataset_wm3[dataset_columnsNames_wm3]

    temp_data_1 = floatDataSplit(temp_data_x1, '密度')
    temp_data_2 = floatDataSplit(temp_data_x2, '含糖率')

    dataset3 = dataset3.join(temp_data_1, how='right', lsuffix='_left', rsuffix='_right')
    dataset3 = dataset3.join(temp_data_2, how='right', lsuffix='_left', rsuffix='_right')

    # 将"好瓜"这一列移动为最后一列
    dataset3.insert(dataset3.shape[1] - 1, '好瓜', dataset3.pop('好瓜'))
    # ==================================================================

    dataset = dataset2
    accelate_rate_list_id3 = []
    accelate_rate_id3 = 0.0
    # print(dataset_columnsNames)

    lengthProgress = 10
    with alive_bar(lengthProgress) as bar:
        for i in range(lengthProgress):
            train_data, test_data = data_split(dataset, 0.9)  # 划分数据集
            # _, test_data = data_split(dataset[dataset_columnsNames], 0.8)  # 划分数据集
            testDataLen = len(test_data)

            decisionTree_ID3 = DecisionTree_ID3()
            treeData = decisionTree_ID3.fit(train_data)
            # print(f"treeData: \n    {treeData}")
            # pt.pprint(treeData)
            # print(f"test_data:\n{test_data}")
            result_id3 = pd.DataFrame({'预测值': decisionTree_ID3.predict(test_data), '正取值': test_data.iloc[:, -1]})
            # print(result_id3)

            accelate_rate_list_id3.append(len(result_id3.query('预测值==正取值')) / testDataLen)
            # 累加成功率
            accelate_rate_id3 += len(result_id3.query('预测值==正取值')) / testDataLen

            bar()
            time.sleep(0.1)

    print('使用ID3作为划分标准时平均准确率为:' + str((accelate_rate_id3 / lengthProgress) * 100) + "%")
