#!/usr/bin/env python
# coding:utf-8
# @TIME         : 2021/4/20 3:36 下午
# @Author       : BTG
# @Project      : pythonProject
# @File Name    : Decision_Tree.py
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
import numpy as np
import pandas as pd


class DecisionTree_ID3:
    def __init__(self):
        self.model = None

    # 计算信息熵
    def calEntropy(self, y):
        """
        :param y:   每条数据对应的类别
        :return:    类别信息熵
        """
        valRate = y.value_counts().apply(lambda x: x / y.size)
        # print(valRate)
        valEntropy = np.inner(valRate, np.log2(valRate)) * -1
        # print(valEntropy)
        return valEntropy

    # 建立ID3决策树
    def buildDecisionTree_ID3(self, xTrain, yTrain):
        """
        :param xTrain:  训练数据
        :param yTrain:  每条数据对应的类别
        :return:        决策树模型
        """
        # 获取属性名
        PropertyNames = xTrain.columns
        # 获取正反例个数
        yTrainCounts = yTrain.value_counts()
        # print(yTrainCounts)

        if yTrainCounts.size == 1:
            # print('only one class', yTrainCounts.index[0])
            return yTrainCounts.index[0]
        # 计算当前样本集合D的信息熵
        entropy_D = self.calEntropy(yTrain)
        # print(f"entropy_D : {entropy_D}")

        maxGain = None  # 最大信息增益
        maxEntropyPropName = None  # 达到最大信息熵的属性
        # 遍历属性
        for propertyName in PropertyNames:
            # 获取属性值
            propDatas = xTrain[propertyName]
            # 频次汇总 得到各个特征对应的概率
            propClassSummary = propDatas.value_counts().apply(lambda x: x / propDatas.size)
            # print(propClassSummary)

            # 遍历频次汇总的结果
            # propClass 各属性取值
            # DvRate = |Dv| / |D|
            # sumEntropyByProp = sum(DvRate * Ent(Dv))  v = 1, 2, ..., V
            sumEntropyByProp = 0
            for propClass, DvRate in propClassSummary.items():
                # 获取对应属性值propClass的所有分类结果
                yDataByPropClass = yTrain[xTrain[propertyName] == propClass]
                # 使用离散属性a来对样本集D进行划分，会产生V个分支结点，第v个分支结点包含了D中所有在属性a上取值为av的样本，记为Dv
                # 计算Dv的信息熵
                entropy_Dv = self.calEntropy(yDataByPropClass)
                sumEntropyByProp += entropy_Dv * DvRate
            # Gain(D, a) <=> gainEach
            gainEach = entropy_D - sumEntropyByProp
            # 获取最大信息增益和属性
            if maxGain == None or gainEach > maxGain:
                maxGain = gainEach
                maxEntropyPropName = propertyName
        # print('select prop:', maxEntropyPropName, maxGain)
        # 获取属性值
        propDatas = xTrain[maxEntropyPropName]
        # 频次汇总 得到各个特征对应的概率
        propClassSummary = propDatas.value_counts().apply(lambda x: x / propDatas.size)

        # ？？？
        retClassByProp = {}
        for propClass, DvRate in propClassSummary.items():
            # 将属性 maxEntropyPropName 的属性值为 propClass 的数据筛选出来
            # print(propClass)
            whichIndex = xTrain[maxEntropyPropName] == propClass
            # 如果不存在 maxEntropyPropName 属性值为 propClass 的数据则查看下一个属性值
            if whichIndex.size == 0:
                continue
            xDataByPropClass = xTrain[whichIndex]
            yDataByPropClass = yTrain[whichIndex]
            del xDataByPropClass[maxEntropyPropName]  # 删除已经选择过的属性列
            # print(pd.concat([xDataByPropClass, yDataByPropClass], axis=1))
            retClassByProp[propClass] = self.buildDecisionTree_ID3(xDataByPropClass, yDataByPropClass)
        return {'Node': maxEntropyPropName, 'Edge': retClassByProp}

    # 为啥叫这个名字？
    def predictBySeries(self, modelNode, data):
        # 判断modelNode是否为字典
        if not isinstance(modelNode, dict):
            # modelNode 不是字典时
            return modelNode
        # modelNode 是字典时
        # 获取根结点值
        nodePropName = modelNode['Node']
        # 从测试数据集一条数据中获取属性为 nodePropName 的属性值 <=> nodePropVal
        nodePropVal = data.get(nodePropName)
        # 遍历结点的边，找到与属性值 nodePropVal 匹配的边
        for edge, nextNode in modelNode['Edge'].items():
            if nodePropVal == edge:
                return self.predictBySeries(nextNode, data)
        return None

    # 决策树训练
    def fit(self, xTrain, yTrain=pd.Series()):
        """
        :param xTrain:  训练数据
        :param yTrain:  每条数据对应的类别
        :return:        训练好的 id3 决策树模型
        """
        if yTrain.size == 0:  # 如果不传，自动选择最后一列作为分类标签
            yTrain = xTrain.iloc[:, -1]
            xTrain = xTrain.iloc[:, :len(xTrain.columns) - 1]

        self.model = self.buildDecisionTree_ID3(xTrain, yTrain)
        return self.model

    # 决策树预测
    def predict(self, data):
        # 这个是干啥的？
        # if isinstance(data, pd.Series):
        #     return self.predictBySeries(self.model, data)
        return data.apply(lambda d: self.predictBySeries(self.model, d), axis=1)


if __name__ == '__main__':
    a = pd.Series()
    d = {'Node':
             '纹理', 'Edge': {'清晰': {'Node':
                                       '根蒂', 'Edge': {'蜷缩': '是',
                                                      '稍蜷': {'Node':
                                                                 '色泽', 'Edge': {'乌黑': {'Node':
                                                                                           '触感', 'Edge': {'硬滑': '是',
                                                                                                          '软粘': '否'}},
                                                                                '青绿': '是'}},
                                                      '硬挺': '否'}},
                            '稍糊': {'Node':
                                       '触感', 'Edge': {'硬滑': '否',
                                                      '软粘': '是'}},
                            '模糊': '否'}}
    print(d['Edge'])
    for edge, nextNode in d['Edge'].items():
        print(edge)
        print(nextNode)
