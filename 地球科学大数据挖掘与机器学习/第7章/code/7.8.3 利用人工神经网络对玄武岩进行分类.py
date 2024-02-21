# -*- coding: utf-8 -*-

# 代码 7-3

import pandas as pd
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers.core import Dense, Activation
#参数初始化
basalt = pd.read_csv('./data/basalt.csv')  
# 浏览数据集  
data = basalt.drop(basalt.columns[1:5],axis=1)
data = data.iloc[:,0:11]
print('转化为布尔值前数据的前4行为：\n',data.head())
data.loc[data['TECTONIC SETTING']=='INTRAPLATE VOLCANICS','TECTONIC SETTING'] = 1
data.loc[data['TECTONIC SETTING']!= 1,'TECTONIC SETTING'] = 0
print('转化为布尔值后数据的前4行为：\n',data.head())
df1 = data.iloc[:,1:11]
def MinMaxScale(data):
    data=(data-data.min())/(data.max()-data.min())
    return data
df2 = MinMaxScale(df1)
x = df2.values
y = data.iloc[:,0].values

model = Sequential() #建立模型
model.add(Dense(input_dim = 10 ,output_dim = 10))
model.add(Activation('relu')) #用relu函数作为激活函数，能够大幅提供准确度
model.add(Dense(input_dim = 10 ,output_dim = 1))
model.add(Activation('sigmoid')) #由于是0-1输出，用sigmoid函数作为激活函数
#另外常见的损失函数还有mean_squared_error、categorical_crossentropy等，请阅读帮助文件。
#求解方法我们指定用adam，还有sgd、rmsprop等可选
model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
model.fit(x, y, nb_epoch = 1000, batch_size = 10) #训练模型，学习一千次
yp = model.predict_classes(x).reshape(len(y)) #分类预测

#自定义混淆矩阵可视化函数
def cm_plot(y, yp):  
  from sklearn.metrics import confusion_matrix #导入混淆矩阵函数  
  cm = confusion_matrix(y, yp) #混淆矩阵   
  plt.matshow(cm, cmap=plt.cm.Greens) #画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。  
  plt.colorbar() #颜色标签  
  for x in range(len(cm)): #数据标签  
    for y in range(len(cm)):  
      plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')  
  plt.ylabel('True label') #坐标轴标签  
  plt.xlabel('Predicted label') #坐标轴标签  
  return plt 
cm_plot(y,yp).show() #显示混淆矩阵可视化结果
cm_plot(y,yp).savefig('./tmp/混淆矩阵.png')
