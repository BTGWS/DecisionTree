#!/usr/bin/env python
# coding:utf-8
# @TIME         : 2021/4/20 3:52 下午
# @Author       : BTG
# @Project      : pythonProject
# @File Name    : Functions.py
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
import math
import time

import numpy as np
import pandas as pd
from alive_progress import alive_bar


# 将数据分成训练集和测试集
def data_split(data: pd.DataFrame, ratio: float):
    """
    :param data:    DataFrame形式的原数据集
    :param ratio:   训练集所占比例
    :return:        分割好的训练集和测试集
    """
    train_num = math.ceil(len(data) * ratio)
    train_data = data.sample(n=train_num, replace=False, axis=0)
    test_data = data.append(train_data).drop_duplicates(keep=False)
    return train_data, test_data


# 将连续值转换成离散值
def floatDataSplit(xTrain, name: str):
    """
    :param xTrain:  连续值的属性值
    :param name:    连续值的属性名
    :return:        离散化的属性值
    """
    data = np.array(xTrain).tolist()
    temp = np.array(xTrain).tolist()

    temp.sort()
    split = []
    for i in range(0, len(temp) - 1):
        split.append((temp[i][0] + temp[i + 1][0]) / 2)
    # 向一个新的dataframe中将连续变量离散化取值
    resultData = pd.DataFrame()
    for i in range(len(split)):
        temp_list = []
        for j in range(len(data)):
            if data[j][0] >= split[i]:
                temp_list.append(1)
            else:
                temp_list.append(0)
        resultData.insert(i, name + '是否大于' + str(split[i]) + '?', temp_list)
    return resultData


def Pbur():
    items = range(100)
    with alive_bar(len(items)) as bar:
        for item in items:
            bar()
            time.sleep(0.1)


if __name__ == '__main__':
    Pbur()
