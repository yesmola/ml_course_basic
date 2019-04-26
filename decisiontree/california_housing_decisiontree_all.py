#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.grid_search import GridSearchCV
from sklearn.tree import export_graphviz
#导入可视化工具
import graphviz
#导入转换 dot 文件的工具
import pydotplus
import os 
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  #注意修改你的路径
#加载数据集
housing = fetch_california_housing()
#取数据集中的所有特征
x=housing.data[:,:]
y=housing.target
#拆分数据集
x_train,x_test,y_train,y_test=train_test_split(x,y)
#初始化决策树，设置决策树的最大深度为 3
decisionTree=DecisionTreeRegressor(max_depth=3)
#拟合训练集
decisionTree.fit(x_train,y_train)
#输出训练集上的评分
print("training set score={}".format(decisionTree.score(x_train,y_train)))

#使用R2,MSE,MAE评测模型
y_predict=decisionTree.predict(x_test)
r2=r2_score(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
mae=mean_absolute_error(y_test,y_predict)
print("r2={}".format(r2))
print("mse={}".format(mse))
print("mae={}".format(mae))


#对决策树回归的参数调优
params={'min_samples_split':[2,3,6,9,12],'max_depth':[3,4,5,6,7]}
dct=decisionTree.fit(x_train,y_train)
gsv=GridSearchCV(dct,params,cv=6).fit(x_train,y_train)
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

