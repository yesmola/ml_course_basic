#!/usr/bin/env python
# coding: utf-8

# In[2]:


#导入SVM回归模型
from sklearn.svm import SVR
#导入拆分数据集工具
from sklearn.model_selection import train_test_split
#数据预处理
from sklearn.preprocessing import StandardScaler,RobustScaler
#下载数据集
from sklearn.datasets import fetch_california_housing

housing=fetch_california_housing()
x,y=housing.data,housing.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

svr=SVR() 
#使用默认参数
svr.fit(x_train,y_train)
svr_predict=svr.predict(x_test)
print("train_score:",svr.score(x_train,y_train))
print("test_score:",svr.score(x_test,y_test))

