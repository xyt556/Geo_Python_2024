# -*- coding: utf-8 -*-

# 代码 3‑1

from sklearn import datasets
from sklearn.cross_decomposition import CCA
import pandas as pd
iris = datasets.load_iris()  # 导入鸢尾花的数据集
iris_x = iris.data[:,0:2]   # 取样本数据前两个特征
iris_y = iris.data[:,2:4]   # 取样本数据后两个特征
print('样本数据前两个特征的前4行数据为：\n',pd.DataFrame(iris_x).head())
print('样本数据后两个特征的前4行数据为：\n',pd.DataFrame(iris_y).head())
cca = CCA()           # 定义一个典型相关分析对象
# 调用该对象的训练方法，主要接收两个参数：两个不同的数据集
cca.fit(iris_x, iris_y)
print('降维结果为：', cca.transform(iris_x, iris_y))  # 输出降维结果