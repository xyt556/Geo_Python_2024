# -*- coding: utf-8 -*-

# 代码 4‑3

# 气象数据离散化预处理
import pandas as pd

weather = pd.read_csv('../data/广州气象数据.csv') #参数初始化
print('离散化前数据的前4行为：\n',weather.head())
weather = weather.dropna(axis = 0,how ='any') #去除空值
# 离散化最高温度
n=10
while n<41:
    for i in range(n-5,n):
        weather.loc[(weather['最高气温']==i),'最高气温'] = '最高气温为%d~%d度' %(n-5,n)
    n+=5
# 离散化最低温度
n=0
while n<31:
    for i in range(n-5,n):
        weather.loc[(weather['最低气温']==i),'最低气温'] = '最低气温为%d~%d度' %(n-5,n)
    n+=5
weather = weather.iloc[:,1:] #去除无用的属性
print('离散化后数据的前4行为：\n',weather.head())
weather.to_csv('./tmp/广州气象数据离散化.csv',encoding = 'gbk',index = False) #保存数据



# 代码4-4

from __future__ import print_function
import pandas as pd
d = pd.read_csv('./tmp/广州气象数据离散化.csv',encoding = 'gbk')
data = d.as_matrix()
print(u'\n转换原始数据至0-1矩阵...')
import time
start = time.clock()

ct = lambda x : pd.Series(1, index = x)
#b = [i for i in map(ct, data)]
b = map(ct, data)
f = list(b)
d = pd.DataFrame(f).fillna(0)

d = (d==1)
end = time.clock()
print(u'\n转换完毕，用时：%0.2f秒' %(end-start))
print(u'\n开始搜索关联规则...')
del b

support = 0.06 #最小支持度
confidence = 0.75 #最小置信度
ms = '--' #连接符，用来区分不同元素，如A--B。需要保证原始表格中不含有该字符

#自定义连接函数，用于实现L_{k-1}到C_k的连接
def connect_string(x, ms):
    x = list(map(lambda i:sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i,len(x)):
            if x[i][:l-1] == x[j][:l-1] and x[i][l-1] != x[j][l-1]:
                r.append(x[i][:l-1]+sorted([x[j][l-1],x[i][l-1]]))
    return r

#寻找关联规则的函数
def find_rule(d, support, confidence):
    import time
    start = time.clock()
    result = pd.DataFrame(index=['support', 'confidence']) #定义输出结果

    support_series = 1.0*d.sum()/len(d) #支持度序列
    column = list(support_series[support_series > support].index) #初步根据支持度筛选
    k = 0

    while len(column) > 1:
        k = k+1
        print(u'\n正在进行第%s次搜索...' %k)
        column = connect_string(column, ms)
        print(u'数目：%s...' %len(column))
        sf = lambda i: d[i].prod(axis=1, numeric_only = True) #新一批支持度的计算函数

        #创建连接数据，这一步耗时、耗内存最严重。当数据集较大时，可以考虑并行运算优化。
        d_2 = pd.DataFrame(list(map(sf,column)), index = [ms.join(i) for i in column]).T

        support_series_2 = 1.0*d_2[[ms.join(i) for i in column]].sum()/len(d) #计算连接后的支持度
        column = list(support_series_2[support_series_2 > support].index) #新一轮支持度筛选
        support_series = support_series.append(support_series_2)
        column2 = []
        
        for i in column: #遍历可能的推理，如{A,B,C}究竟是A+B-->C还是B+C-->A还是C+A-->B？
            i = i.split(ms)
            for j in range(len(i)):
                column2.append(i[:j]+i[j+1:]+i[j:j+1])
        
        cofidence_series = pd.Series(index=[ms.join(i) for i in column2]) #定义置信度序列
        
        for i in column2: #计算置信度序列
            cofidence_series[ms.join(i)] = support_series[ms.join(sorted(i))]/support_series[ms.join(i[:len(i)-1])]
        
        for i in cofidence_series[cofidence_series > confidence].index: #置信度筛选
            result[i] = 0.0
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort_values(['confidence','support'], ascending = False) #结果整理，输出
    end = time.clock()
    print(u'\n搜索完成，用时：%0.2f秒' %(end-start))
    print(u'\n结果为：')
    print(result)
    
    return result

find_rule(d, support, confidence).to_excel('./tmp/rules.xls')