# -*- coding: utf-8 -*-

# 代码 5-3

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import networkx as nx 
# 读取数据
data = pd.read_csv('./data/basalt.csv')
data.drop(['SAMPLE NAME'],axis=1,inplace=True)
association = data.corr()
# 数据预处理
# 筛选出相似的属性
delSimCol = []
colNum = association.shape[0]
names = association.columns
for i in range(colNum):
    for j in range(i+1,colNum):
        if association.iloc[i,j]>0.9:
            delSimCol.append((names[i],names[j]))
print('经过筛选得到的相似的属性为：\n',delSimCol)  
delCol = [i[1] for i in delSimCol] 
data.drop(delCol,axis=1,inplace = True) # 删除列
dummiesData = pd.get_dummies(data['LAND OR SEA']) # 哑变量处理
data.drop('LAND OR SEA',axis=1,inplace=True)

# 绘制轮廓系数柱形图
modelData = pd.concat([data,dummiesData],axis=1)
x = modelData.iloc[:,1:]
X = StandardScaler().fit_transform(x)
silhouettteScore = []
for k in range(2,18):
    kmeans = KMeans(n_clusters=k, random_state=123).fit(X)
    score = silhouette_score(X,kmeans.labels_)
    silhouettteScore.append(score)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False ## 设置正常显示符号
plt.figure(figsize=(10,6))
plt.plot(range(2,18),silhouettteScore)
plt.bar(range(2,18),silhouettteScore)
plt.xlabel('聚类数目')
plt.ylabel('轮廓系数')
plt.savefig('./tmp/轮廓系数.png')
plt.show()

# 绘制雷达图
kmeans = KMeans(n_clusters=9, random_state=123).fit(X)
label = kmeans.labels_
data['cluster_label'] = label
# data.to_csv('./tmp/聚类结果.csv',index=False)
center = kmeans.cluster_centers_
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, polar=True)# polar参数
angles = np.linspace(0, 2*np.pi, 31, endpoint=True)
names = x.columns
for i in range(9):
    Data = np.concatenate((center[i], [center[i][0]])) # 闭合
    ax.plot(angles,Data, linewidth=2)# 画线
    ax.fill(angles, Data, alpha=0.25)# 填充    
ax.set_thetagrids(angles * 180/np.pi, names)
ax.set_title("聚类结果雷达图", va='bottom')## 设定标题
ax.set_rlim(-1,2.5)## 设置各指标的最终范围
ax.legend(range(9),loc=0)
ax.grid(True)
plt.savefig('./tmp/聚类结果雷达图.png')

# 绘制网络图
header = data.iloc[:,0]
tailer = label
G = nx.Graph()
for i in range(data.shape[0]):
    head,tail = header[i],tailer[i]
    G.add_edge(head,tail)
klist = list(nx.algorithms.community.k_clique_communities(G,2))
plt.figure(figsize=(15,15)) #创建一幅图
nx.draw(G, node_color='red',nodelist = klist[0], with_labels=True, node_size=800)
plt.savefig('./tmp/社区发现.png')
plt.show()


