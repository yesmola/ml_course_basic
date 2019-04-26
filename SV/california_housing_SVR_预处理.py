#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

#预处理
rb=RobustScaler().fit(x_train)
x_train=rb.transform(x_train)
x_test=rb.transform(x_test)
ss=StandardScaler(with_mean=True,with_std=True).fit(x_train)
x_train=ss.transform(x_train)
x_test=ss.transform(x_test)

svr=SVR() 
#使用默认参数
svr.fit(x_train,y_train)
svr_predict=svr.predict(x_test)
print("train_score:",svr.score(x_train,y_train))
print("test_score:",svr.score(x_test,y_test))

#调参
svr_new=SVR(C=9,gamma=0.3,degree=3) #调整后的参数
svr_new.fit(x_train,y_train)
svr_predict=svr_new.predict(x_test)
print("new_train_score:",svr_new.score(x_train,y_train))
print("new_test_score:",svr_new.score(x_test,y_test))

