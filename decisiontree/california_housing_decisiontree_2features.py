#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.tree import export_graphviz
#导入可视化工具
import graphviz
#导入转换 dot 文件的工具
import pydotplus
#修改系统环境变量bu
import os 
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  #注意修改你的路径
#加载数据集
housing = fetch_california_housing()
#取数据集中的前 2 个特征
x=housing.data[:,:2]
y=housing.target
#拆分数据集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
#初始化决策树回归器，设置决策树的最大深度为 3
decisionTree=DecisionTreeRegressor(max_depth=3)
#拟合训练集
decisionTree.fit(x_train,y_train)
#输出训练集上的评分
print(decisionTree.score(x_train,y_train))
#将决策树可视化，即用图来表示
dot_data=export_graphviz(decisionTree,out_file=None,feature_names=housing.feature_names[:2],
impurity=False,rounded=True,filled=True)
#将 dot 类型的数据转换为图
graph = pydotplus.graph_from_dot_data(dot_data)
#保存为图片格式
graph.write_png('c:\\yuanye\\house.png')


# In[ ]:




