# -*- coding: utf-8 -*-

# 代码 6-6

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine

# 读取数据文件
weather = pd.read_csv('../data/广州气象数据.csv')
xtick = weather.iloc[:,0]
weather = weather.iloc[:,[1,2]]

temp = weather.values.T
high = temp[0,:]
low = temp[1,:]

# 计算最低最高气温的余弦相似度
print('相似度为：',1-cosine(high,low))

# 绘制气温变化趋势曲线
# 利用多项式拟合对数据进行平滑，使曲线更加简明
length_data = len(high)+1
x = np.arange(1, length_data, 1)
z1 = np.polyfit(x, high, 50)#50拟合
z2 = np.polyfit(x, low, 50)#50拟合
p1 = np.poly1d(z1)
p2 = np.poly1d(z2)
high2 = p1(x)
low2 = p2(x)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False ## 设置正常显示符号
plt.figure(figsize=(9,9))
plt.ylabel("temperature(℃)")
plt.xlabel("days")
plt.xticks(range(1,len(high2),100),xtick[0:len(high2):100],rotation=60)
plt.plot(high2,linewidth = 1,color='black',label="最高气温")
plt.plot(low2,linewidth = 1,color='black',linestyle = '--',label = "最低气温")
plt.legend()
plt.savefig('../tmp/折线图1',dpi=220)