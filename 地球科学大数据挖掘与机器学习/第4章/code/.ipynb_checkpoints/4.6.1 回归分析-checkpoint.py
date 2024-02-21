# -*- coding: utf-8 -*-

# 代码 4‑1

# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
# basalt = pd.read_csv('./data/basalt.csv')
#
# # 删除无用数据
# df = basalt.drop(basalt.columns[1:5],axis=1)
# print('转化为布尔值前数据的前4行为：\n',df.head())
# df.loc[df['TECTONIC SETTING']=='INTRAPLATE VOLCANICS','TECTONIC SETTING'] = 1
# df.loc[df['TECTONIC SETTING']!= 1,'TECTONIC SETTING'] = 0
# print('转化为布尔值后数据的前4行为：\n',df.head())
#
# X_train, X_test, y_train, y_test = train_test_split(df.ix[:, 1:], df.ix[:, 0], test_size=.1, random_state=520)
# lr = LogisticRegression()    # 建立LR模型
# lr.fit(X_train, y_train)    # 用处理好的数据训练模型
# print ('逻辑回归的准确率为：{0:.2f}%'.format(lr.score(X_test,y_test)*100))


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

basalt = pd.read_csv('../data/basalt.csv')

# 删除无用数据
df = basalt.drop(basalt.columns[1:5], axis=1)
print('转化为布尔值前数据的前4行为：\n', df.head())

# 使用 map 方法将标签转换为整数
df['TECTONIC SETTING'] = df['TECTONIC SETTING'].map({'INTRAPLATE VOLCANICS': 1, '其他标签值': 2})  # 添加其他标签值

print('转化为布尔值后数据的前4行为：\n', df.head())

# 删除包含NaN的行
df = df.dropna()

# 使用 iloc 获取训练和测试数据
X = df.iloc[:, 1:]  # 特征
y = df.iloc[:, 0]  # 标签

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=520)
lr = LogisticRegression()  # 建立LR模型
lr.fit(X_train, y_train)  # 用处理好的数据训练模型
print('逻辑回归的准确率为：{0:.2f}%'.format(lr.score(X_test, y_test) * 100))

