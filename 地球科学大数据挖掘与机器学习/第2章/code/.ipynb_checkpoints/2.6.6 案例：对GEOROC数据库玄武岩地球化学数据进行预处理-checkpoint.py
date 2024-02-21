# -*- coding: utf-8 -*-

# 代码2-6

import pandas as pd
basalt = pd.read_csv('./data/basalt_pre.csv')
#数据清洗
print("清洗前的数据的形状为：",basalt.shape)
basalt = basalt.loc[(basalt['SIO2(WT%)']>45)&(basalt['SIO2(WT%)']<55),:]
basalt = basalt.loc[(basalt['TIO2(WT%)']>0.1),:]
basalt = basalt.loc[(basalt['AL2O3(WT%)']>10)&(basalt['AL2O3(WT%)']<18),:]
basalt = basalt.loc[(basalt['H2OT(WT%)']<7)|(basalt['H2OT(WT%)'].isnull()),:]
basalt = basalt.loc[(basalt['LOI(WT%)']<7)|(basalt['LOI(WT%)'].isnull()),:]
basalt = basalt.loc[(basalt['CO2(WT%)']<3)|(basalt['CO2(WT%)'].isnull()),:]
basalt.drop(labels = ['H2OT(WT%)','CO2(WT%)','LOI(WT%)'],axis = 1,inplace = True)
print("清洗后的数据的形状为：",basalt.shape)

#查看每个属性的缺失值数量
print('各列的缺失值数量为：\n', basalt.isnull().sum())
print('去除缺失值前的数据的形状为：', basalt.shape)
print('去除缺失值后的数据的形状为：',basalt.dropna(axis = 0,how ='any').shape)
#删除缺失值
basalt = basalt.dropna(axis = 0,how ='any')

#定义异常值处理函数
def outRange(Ser1):
    QL = Ser1.quantile(0.1)
    QU = Ser1.quantile(0.9)
    IQR = QU-QL
    Ser1.loc[Ser1>(QU+1.5*IQR)] = None
    Ser1.loc[Ser1<(QL-1.5*IQR)] = None
    return Ser1
#异常值处理
df = basalt.copy()
names = df.columns
for j in names[7:]:
    df[j] = outRange(df[j])
print('各列数据的异常值数量为：\n',df.isnull().sum())
df.dropna(axis = 0,how ='any',inplace=True)
print('去除异常值前的数据的形状为：', basalt.shape)
print('去除异常值后的数据的形状为：',df.shape)
#保存数据    
df.to_csv('./tmp/basalt.csv',index = False)