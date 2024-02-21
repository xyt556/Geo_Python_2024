# -*- coding: utf-8 -*-

# 代码 6-1

import pandas as pd
#导入数据表，并显示前5行
data = pd.read_csv('./data/广州气象数据.csv')
#将“日期”一列转为日期格式，在此基础上才能进行日期的范围查询
data['日期'] = pd.to_datetime(data['日期'])
data.head()



# 代码 6-2

# 查询
def select(DataFrame,date= 'all',max_temperature='all',min_temperature= 'all',weather='all',wind_destination= 'all',wind_leval= 'all'):
    if (date != 'all') & isinstance(date,str):
        exp0 = DataFrame.iloc[:,0] == date
    elif isinstance(date,list):
        exp0 = (DataFrame.iloc[:,0]>=date[0])&(DataFrame.iloc[:,0]<=date[1])
    else:
        exp0 = True
    if max_temperature != 'all':
        exp1 = DataFrame.iloc[:,1] == max_temperature
    else:
        exp1 = True
    if min_temperature != 'all':
        exp2 = DataFrame.iloc[:,2] == min_temperature
    else:
        exp2 = True
    if weather != 'all':
        exp3 = DataFrame.iloc[:,3] == weather
    else:
        exp3 = True
    if wind_destination != 'all':
        exp4 = DataFrame.iloc[:,4] == wind_destination
    else:
        exp4 = True
    if wind_leval != 'all':
        exp5 = DataFrame.iloc[:,5] == wind_leval
    else:
        exp5 = True
    exp = exp0&exp1&exp2&exp3&exp4&exp5
    return DataFrame.loc[exp,:]



# 代码 6-3

# 查询data数据中日期为2015年1月1日的气象数据
select(data, date='2015-01-01')



# 代码 6-4

# 查询data数据中日期为2015年1月12日至2015年1月18日的气象数据
select(data,date=['2015-1-12','2015-1-18'])



# 代码 6-5

import pandas as pd
import numpy as np
def percentail(DataFrame,percent = [5, 25, 50, 75]):
    names = DataFrame.columns[DataFrame.dtypes == 'int64']
    indexs = [str(i)+'%分位数' for i in percent]
    df = pd.DataFrame(columns=names,index=indexs)
    for name in names:
        df[name] = np.percentile(DataFrame[name],percent)
    return df
percentail(data)