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
import json
import pandas as pd


from alive_progress import alive_bar

from IOFiles import read_file
from Functions import data_split, floatDataSplit
from Decision_Tree import DecisionTree_ID3, DecisionTree_C45
import pprint as pt

"""    
    print(dataset.columns.values)   # 通过columns字段获取，返回一个numpy型的array
    print(list(dataset))            # 直接使用 list 关键字，返回一个list
    print(dataset.columns.tolist()) # df.columns 返回Index，可以通过 tolist(), 或者 list（array） 转换为list
"""
start_head = 'dataset/'
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
    accelerate_rate_list_id3 = []
    accelerate_rate_list_c45 = []
    accelerate_rate_id3 = 0.0
    accelerate_rate_c45 = 0.0
    # print(dataset_columnsNames)

    lengthProgress = 10
    train_ratio = 0.8
    with alive_bar(lengthProgress) as bar:
        train_data, _ = data_split(dataset, train_ratio)  # 划分数据集

        for i in range(lengthProgress):
            _, test_data = data_split(dataset, train_ratio)  # 划分数据集
            testDataLen = len(test_data)
            # print(test_data)

            # ================ID3======================
            decisionTree_ID3 = DecisionTree_ID3()
            treeData_ID3 = decisionTree_ID3.fit(train_data)  # treeData_ID3为训练好的id3模型
            # print(f"treeData: \n    {treeData}")
            # pt.pprint(treeData)
            # print(f"test_data:\n{test_data}")
            result_id3 = pd.DataFrame({'预测值': decisionTree_ID3.predict(test_data), '正取值': test_data.iloc[:, -1]})
            # print(result_id3)
            accelerate_rate_list_id3.append(len(result_id3.query('预测值==正取值')) / testDataLen)
            # 累加成功率
            accelerate_rate_id3 += len(result_id3.query('预测值==正取值')) / testDataLen

            print(json.dumps(treeData_ID3, ensure_ascii=False))
            # ================ID3======================

            # ================C4.5======================
            decisionTree_C45 = DecisionTree_C45()
            treeData_C45 = decisionTree_C45.fit(train_data)
            result_c45 = pd.DataFrame({'预测值': decisionTree_C45.predict(test_data), '正取值': test_data.iloc[:, -1]})
            accelerate_rate_list_c45.append(len(result_c45.query('预测值==正取值')) / testDataLen)
            # 累加成功率
            accelerate_rate_c45 += len(result_c45.query('预测值==正取值')) / testDataLen

            print(json.dumps(treeData_C45, ensure_ascii=False))
            # ================C4.5======================

            bar()
            time.sleep(0.1)

    print('使用ID3作为划分标准时平均准确率为:' + str((accelerate_rate_id3 / lengthProgress) * 100) + "%")
    print('使用C4.5作为划分标准时平均准确率为:' + str((accelerate_rate_c45 / lengthProgress) * 100) + "%")
