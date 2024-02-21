# -*- coding: utf-8 -*-

# 代码 4‑5

import numpy as np
import pandas as pd
import math
def prediction(df,userdf,Nn=15):#Nn邻居个数
    corr=df.T.corr()
    rats=userdf.copy()
    for usrid in userdf.index:
        dfnull=df.loc[usrid][df.loc[usrid].isnull()]
        usrv=df.loc[usrid].mean()#评价平均值
        for i in range(len(dfnull)):
            nft=(df[dfnull.index[i]]).notnull()
            #获取邻居列表
            if(Nn<=len(nft)):
                nlist=df[dfnull.index[i]][nft][:Nn]
            else:
                nlist=df[dfnull.index[i]][nft][:len(nft)]
            nlist=nlist[corr.loc[usrid,nlist.index].notnull()]
            nratsum=0
            corsum=0
            if(0!=nlist.size):
                nv=df.loc[nlist.index,:].T.mean()#邻居评价平均值
                for index in nlist.index:
                    ncor=corr.loc[usrid,index]
                    nratsum+=ncor*(df[dfnull.index[i]][index]-nv[index])
                    corsum+=abs(ncor)
                if(corsum!=0):
                    rats.at[usrid,dfnull.index[i]]= usrv + nratsum/corsum
                else:
                    rats.at[usrid,dfnull.index[i]]= usrv
            else:
                rats.at[usrid,dfnull.index[i]]= None
    return rats
def recomm(df,userdf,Nn=15,TopN=3):
    ratings=prediction(df,userdf,Nn)#获取预测评分
    recomm=[]#存放推荐结果
    for usrid in userdf.index:
        #获取按NA值获取未评分项
        ratft=userdf.loc[usrid].isnull()
        ratnull=ratings.loc[usrid][ratft]
        #对预测评分进行排序
        if(len(ratnull)>=TopN):
            sortlist=(ratnull.sort_values(ascending=False)).index[:TopN]
        else:
            sortlist=ratnull.sort_values(ascending=False).index[:len(ratnull)]
        recomm.append(sortlist)
    return ratings,recomm



# 代码 4‑6

#使用基于UBCF算法对电影进行推荐
from __future__ import print_function
import pandas as pd

############    主程序   ##############
if __name__ == "__main__":
    print("\n--------------使用基于UBCF算法对电影进行推荐 运行中... -----------\n")
    traindata = pd.read_csv('../data/u1.base',sep='\t', header=None,index_col=None)
    testdata = pd.read_csv('../data/u1.test',sep='\t', header=None,index_col=None)
    #删除时间标签列
    traindata.drop(3,axis=1, inplace=True)
    testdata.drop(3,axis=1, inplace=True)
    #行与列重新命名
    traindata.rename(columns={0:'userid',1:'movid',2:'rat'}, inplace=True)
    testdata.rename(columns={0:'userid',1:'movid',2:'rat'}, inplace=True)
    traindf=traindata.pivot(index='userid', columns='movid', values='rat')
    testdf=testdata.pivot(index='userid', columns='movid', values='rat')
    traindf.rename(index={i:'usr%d'%(i) for i in traindf.index} , inplace=True)
    traindf.rename(columns={i:'mov%d'%(i) for i in traindf.columns} , inplace=True)
    testdf.rename(index={i:'usr%d'%(i) for i in testdf.index} , inplace=True)
    testdf.rename(columns={i:'mov%d'%(i) for i in testdf.columns} , inplace=True)
    userdf=traindf.loc[testdf.index]
    #获取预测评分和推荐列表
    trainrats,trainrecomm=recomm(traindf,userdf)



# 代码 4-7

def InitStat(records):  
    user_tags = dict()  
    tag_items = dict()  
    user_items = dict()  
    for user, item, tag in records.items():  
        addValueToMat(user_tags, user, tag, 1)  
        addValueToMat(tag_items, tag, item, 1)  
        addValueToMat(user_items, user, item, 1)



# 代码 4-8

def Recommend(user):  
    recommend_items = dict()  
    tagged_items = user_items[user]  
    for tag, wut in user_tags[user].items():  
        for item, wti in tag_items[tag].items():  
            #if items have been tagged, do not recommend them  
            if item in tagged_items:  
                continue  
            if item not in recommend_items:  
                recommend_items[item] = wut * wti  
            else:  
                recommend_items[item] += wut * wti  
    return recommend_items



# 代码 4-9

#计算余弦相似度  
def CosineSim(item_tags,i,j):  
    ret = 0  
    for b,wib in item_tags[i].items():     #求物品i,j的标签交集数目  
        if b in item_tags[j]:  
            ret += wib * item_tags[j][b]  
    ni = 0  
    nj = 0  
    for b, w in item_tags[i].items():      #统计 i 的标签数目  
        ni += w * w  
    for b, w in item_tags[j].items():      #统计 j 的标签数目  
        nj += w * w  
    if ret == 0:  
        return 0  
    return ret/math.sqrt(ni * nj)          #返回余弦值



 # 代码 4-10

 #计算推荐列表多样性  
def Diversity(item_tags,recommend_items):  
    ret = 0  
    n = 0  
    for i in recommend_items.keys():  
        for j in recommend_items.keys():  
            if i == j:  
                continue  
            ret += CosineSim(item_tags,i,j)  
            n += 1  
    return ret/(n * 1.0)  



# 代码 4-11

def RecommendPopularTags(user,item, tags, N):  
    return sorted(tags.items(), key=itemgetter(1), reverse=True)[0:N]



# 代码 4 -12

def RecommendItemPopularTags(user,item, item_tags, N):
    return sorted(item_tags[item].items(), key=itemgetter(1), reverse=True)[0:N] 



# 代码 4-13

def RecommendUserPopularTags(user,item, user_tags, N):
    return sorted(user_tags[user].items(), key=itemgetter(1), reverse=True)[0:N] 



# 代码 4-14

def RecommendHybridPopularTags(user,item, user_tags, item_tags, alpha, N):  
    max_user_tag_weight = max(user_tags[user].values())  
    for tag, weight in user_tags[user].items():  
        ret[tag] = (1 – alpha) * weight / max_user_tag_weight  
    max_item_tag_weight = max(item_tags[item].values())  
    for tag, weight in item_tags[item].items():  
        if tag not in ret:  
            ret[tag] = alpha * weight / max_item_tag_weight  
        else:  
            ret[tag] += alpha * weight / max_item_tag_weight  
    return sorted(ret[user].items(), key=itemgetter(1), reverse=True)[0:N]
