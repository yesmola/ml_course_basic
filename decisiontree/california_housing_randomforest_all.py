#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.grid_search import GridSearchCV
#加载数据集
housing = fetch_california_housing()
#取数据集中的特征
x=housing.data[:,:]
y=housing.target
#拆分数据集
x_train,x_test,y_train,y_test=train_test_split(x,y)
#初始化随机森林，设置决策树的最大深度为 3
rfg=RandomForestRegressor(max_depth=3,random_state=10)
#拟合训练集
rfg.fit(x_train,y_train)
#输出训练集上的评分
print("training set score={}".format(rfg.score(x_train,y_train)))

#使用R2,MSE,MAE评测模型
y_predict=rfg.predict(x_test)
r2=r2_score(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
mae=mean_absolute_error(y_test,y_predict)
print("r2={}".format(r2))
print("mse={}".format(mse))
print("mae={}".format(mae))

#对随机森林回归的参数调优
params={'min_samples_leaf':[9,10,11,12,13],'n_estimators':[10,11,12]}
rf=rfg.fit(x_train,y_train)
gsv=GridSearchCV(rf,params,cv=6).fit(x_train,y_train)
print("after optimizing...")
print("best_param:",gsv.best_params_)
print("best_score:",gsv.best_score_)
#使用最优模型进行测试并评价结果
best_model=gsv.best_estimator_
y_predict=best_model.predict(x_test)
r2=r2_score(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
mae=mean_absolute_error(y_test,y_predict)
print("r2={}".format(r2))
print("mse={}".format(mse))
print("mae={}".format(mae))

