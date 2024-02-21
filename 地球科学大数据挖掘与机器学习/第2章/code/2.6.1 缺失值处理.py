# -*- coding: utf-8 -*-

# 代码2-1

import numpy as np
import scipy
x=np.array([1,2,3,4,5,8,9,10])            # 创建自变量x
y1=np.array([2,8,18,32,50,128,162,200])   # 创建因变量y1
y2=np.array([3,5,7,9,11,17,19,21])       # 创建因变量y2
Lag_Value1 = scipy.interpolate.lagrange(x,y1)  # Lagrange插值拟合x,y1
Lag_Value2 = scipy.interpolate.lagrange(x,y2) # Lagrange插值拟合x,y2
print('当x为6,7时，使用Lagrange插值y1为：', Lag_Value1([6,7]))
print('当x为6,7时，使用Lagrange插值y2为：', Lag_Value2([6,7]))


# 代码2‑2

# 自定义一阶跳跃差分函数
def diff_self (xi,k):
    '''
    xi：接收array。表示自变量x。无默认，不可省略。
    k：接收int。表示差分的次数，无默认，不可省略
    '''
    diffValue = []
    for i in range(len(xi)-k):
        diffValue.append(xi[i+k]-xi[i])
    return diffValue
# 自定义求取差商函数
def diff_quot(xi,yi):
    '''
    xi：接收array。表示自变量x。无默认，不可省略。
    yi：接收array。表示因变量y。无默认，不可省略。
    '''
    length = len(xi)
    quot = []
    temp = yi
    for i in range(1,length):
        tem = np.diff(temp,1)/diff_self(xi,i)# 此处需要numpy广播特性支持
        quot.append(tem[0])
        temp = tem
    return(quot)
# 自定义求取(x-x0)*(x-x1).....*(x-x0)
def get_Wi(k = 0, xi = []):
    '''
    xi：接收array。表示自变量x。无默认，不可省略。
    yi：接收array。表示因变量y。无默认，不可省略。
    '''
    def Wi(x):
        '''
        x：接收int，float，ndarray。表示插值节点。无默认。
        '''
        result = 1.0
        for each in range(k):
            result *= (x - xi[each])
        return result
    return Wi
# 自定义牛顿插值公式
def get_Newton_inter(xi,yi):
    '''
    xi：接收array。表示自变量x。无默认，不可省略。
    yi：接收array。表示因变量y。无默认，不可省略。
    '''
    diffQuot = diff_quot(xi,yi)
    def Newton_inter(x):
        '''
        x：接收int，float，ndarray。表示插值节点。无默认。
        '''
        result = yi[0]
        for i in range(0, len(xi)-1):
            result += get_Wi(i+1,xi)(x)*diffQuot[i]
        return result
    return Newton_inter
Newt_Value1 = get_Newton_inter(x,y1)  # Newton插值拟合x,y1
Newt_Value2 = get_Newton_inter(x,y2) # Newton插值拟合x,y2
print('当x为6,7时，使用Newton插值y1为：', Newt_Value1([6,7]))
print('当x为6,7时，使用Newton插值y2为：', Newt_Value2([6,7]))