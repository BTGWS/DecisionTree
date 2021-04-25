#!/usr/bin/env python
# coding:utf-8
# @TIME         : 2021/4/20 3:17 下午
# @Author       : BTG
# @Project      : pythonProject
# @File Name    : IOFiles.py
"""
# Code is far away from bugs with the god animal protecting.
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃        ┃
            ┃  ┳┛  ┗┳  ┃:
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""

import pandas as pd


def read_file(filename):
    temp = filename.split('/')
    temp = temp[-1].split('.')
    dataset = []
    # print(temp)

    if temp[-1] == 'csv':
        dataset = pd.read_csv(filename)
    else:
        dataset = pd.read_excel(filename)
    return dataset


def writeFile(dataset, filename):
    """
    :param dataset:     需要存储的数据集
    :param filename:    文件名
    :return:
    """
    writer = pd.ExcelWriter(filename + '.xlsx')  # 写入Excel文件
    dataset.to_excel(writer, 'page_1', float_format='%.5f', header=True, index=False)  # ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()
