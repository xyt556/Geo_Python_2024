# -*- coding: utf-8 -*-

# 代码2‑4

def SameRateCut(data,k):
    w=data.quantile(np.arange(0,1+1.0/k,1.0/k))
    data=pd.cut(data,w)
    return data


# 代码2‑5
from sklearn.cluster import KMeans #引入KMeans
##自定义数据k-Means聚类离散化函数
def KmeanCut(data,k):

    kmodel=KMeans(n_clusters=k,n_jobs=4)   ##建立模型，n_jobs是并行数
    kmodel.fit(data.reshape((len(data), 1)))    ##训练模型
    c=pd.DataFrame(kmodel.cluster_centers_).sort_values(0)   ##输出聚类中心并排序
    w=pd.rolling_mean(c, 2).iloc[1:]    ##相邻两项求中点，作为边界点
    w=[0]+list(w[0])+[data.max()]    ##把首末边界点加上
    data=pd.cut(data,w)
    return data
