#!/usr/bin/env python
# coding: utf-8

# In[1]:


#导入线性回归模型
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

#1.数据准备
housing = fetch_california_housing()
data=housing.data
target=housing.target
#2.将数据拆分为训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=1)
#3.初始化LR函数
lr=LinearRegression()
#4.使用LR拟合训练集，得到预测模型
predict_model=lr.fit(x_train,y_train)

#从预测模型中获得直线系数和截距
coef=predict_model.coef_            #直线系数
intercept=predict_model.intercept_  #截距
print("coef={}".format(coef))
print("intercept={}".format(intercept))

#使用R2,MSE,MAE评测模型
y_predict=predict_model.predict(x_test)
r2=r2_score(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
mae=mean_absolute_error(y_test,y_predict)
print("r2={}".format(r2))
print("mse={}".format(mse))
print("mae={}".format(mae))





