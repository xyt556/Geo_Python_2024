# -*- coding: utf-8 -*-

# 代码 4‑2

import pandas as pd 
import scipy.cluster.vq as vq
import matplotlib.pylab as plt
basalt = pd.read_csv('./data/basalt.csv')
df = basalt.iloc[:,5:15]
data1 = vq.whiten(df)  # 白化数据

kmeans_cent2 = vq.kmeans(data1, 6)
print('聚类中心为：\n', kmeans_cent2[0])

# 画出单位化后和聚类中心的散点图
p = plt.figure(figsize=(81,81))
for i in range(9):
    for j in range(9):
        ax = p.add_subplot(9,9,i*9+1+j)
        plt.scatter(data1[:, j], data1[:, i])
        plt.scatter(kmeans_cent2[0][:, j], kmeans_cent2[0][:, i], c='r')
plt.savefig('./tmp/聚类结果.png')
plt.show()