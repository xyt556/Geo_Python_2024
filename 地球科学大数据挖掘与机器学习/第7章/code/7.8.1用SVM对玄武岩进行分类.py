# -*- coding: utf-8 -*-

# 代码 7-1

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
basalt = pd.read_csv('./data/basalt.csv')
## 将数据和标签拆开
basalt_data = basalt.iloc[:,5:]
basalt_target = basalt.iloc[:,0]

## 划分训练集，测试集
basalt_train,basalt_test,basalt_target_train,basalt_target_test = \
train_test_split(basalt_data,basalt_target,train_size = 0.8,random_state = 42)

## 标准化
stdScaler = StandardScaler().fit(basalt_train)
basalt_std_train = stdScaler.transform(basalt_train)
basalt_std_test = stdScaler.transform(basalt_test)

print('数据的前4行为：\n',basalt_data.head())
print('目标标签的前4行为：\n',basalt_target.head())

## 建模
svm_basalt = SVC().fit(basalt_std_train,basalt_target_train)
print('建立的SVM模型为：','\n',svm_basalt)

basalt_target_pred = svm_basalt.predict(basalt_std_test)
print('basalt数据集的SVM分类报告为：\n',
      classification_report(basalt_target_test,basalt_target_pred))