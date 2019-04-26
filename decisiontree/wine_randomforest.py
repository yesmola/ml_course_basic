#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import r2_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
#加载数据集
wine=load_wine()
#取数据集中的所有特征
x=wine.data[:,:]
y=wine.target
#拆分数据集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.45,random_state=10)
#初始化决策树分类器，设置决策树分类器的最大深度为 3
rfc=RandomForestClassifier(max_depth=3,random_state=10)
#拟合训练集
rfc.fit(x_train,y_train)
#输出训练集上的分类评分
print("training set score={}".format(rfc.score(x_train,y_train)))
#输出测试集上的评分
print("test set score={}".format(rfc.score(x_test,y_test)))
#使用R2评测模型
y_predict=rfc.predict(x_test)
r2=r2_score(y_test,y_predict)
print("r2={}".format(r2))

#对随机森林回归的参数调优
params={'min_samples_leaf':[2,3,4,5,6],'n_estimators':[10,11,12]}
rf=rfc.fit(x_train,y_train)
gsv=GridSearchCV(rf,params,cv=6).fit(x_train,y_train)
print("after optimizing...")
print("best_param:",gsv.best_params_)
print("best_score:",gsv.best_score_)
#使用最优模型进行测试并评价结果
best_model=gsv.best_estimator_
y_predict=best_model.predict(x_test)
r2=r2_score(y_test,y_predict)
print("r2={}".format(r2))

