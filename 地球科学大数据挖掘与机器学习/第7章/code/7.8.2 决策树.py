# -*- coding: utf-8 -*-

# 代码 7-2

import pydotplus
import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier as DTC
import numpy as np
#参数初始化
data = pd.read_excel('./data/susceptibility.xls', index_col = u'index') #导入数据
#数据是类别变量，要将它转换为1、-1数值变量
#用1来表示“strong”、“rich”、“likely”这三个属性，用-1来表示“weak”、“poor”、“unlikely”
print('数据变化前:\n',data.head())
data[data == 'strong'] = 1
data[data == 'rich'] = 1
data[data == 'likely'] = 1
data[data != 1] = -1
print('数据变化后:\n',data.head())
x = data.iloc[:,:3].astype(int)
y = data.iloc[:,3].astype(int)

dtc = DTC(criterion='entropy') #建立决策树模型，基于信息熵
dtc.fit(x, y) #训练模型
#导入相关函数，可视化决策树。
#导出的结果是一个dot文件，需要安装Graphviz才能将它转换为pdf或png等格式。
data_feature_name = data.columns[:3]
data_target_name = ['likely','unlikely']
f = export_graphviz(dtc,out_file=None, feature_names=data_feature_name,
                    class_names=data_target_name,  filled=True, rounded=True,
                    special_characters=True)
graph = pydotplus.graph_from_dot_data(f)

# 保存图像到pdf文件
# 需要在http://www.graphviz.org/下载相应Graphviz版本
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
graph.write_pdf("./tmp/susceptibility.pdf")