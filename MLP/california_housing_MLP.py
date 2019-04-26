#!/usr/bin/env python
# coding: utf-8

# In[24]:


from sklearn.neural_network import MLPRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn import preprocessing
import numpy as np
housing=fetch_california_housing()
x=housing.data
y=housing.target
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.2)

#数据标准化
ss_x=preprocessing.StandardScaler().fit(x_train)
x_train=ss_x.transform(x_train)
x_test=ss_x.transform(x_test)

#多层感知机回归模型
mlp=MLPRegressor(random_state=0)
mlp.fit(x_train,y_train)
mlp_score=mlp.score(x_test,y_test)
print('回归模型得分',mlp_score)

#使用R2,MSE,MAE评测模型
y_predict=mlp.predict(x_test)
r2=r2_score(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
mae=mean_absolute_error(y_test,y_predict)
print("r2={}".format(r2))
print("mse={}".format(mse))
print("mae={}".format(mae))


# In[51]:


#多层感知机回归模型调参优化
#mlp=MLPRegressor(solver='lbfgs',hidden_layer_sizes = (40,40),activation = 'relu',alpha = 1e-3)
#mlp=MLPRegressor(solver='lbfgs',activation = 'tanh',random_state=0)
#mlp=MLPRegressor(solver='lbfgs',activation = 'relu',hidden_layer_sizes = (20,20,20),random_state=0)
#mlp=MLPRegressor(solver='lbfgs',activation = 'relu',hidden_layer_sizes = (100,100,100),random_state=0)
mlp=MLPRegressor(solver='lbfgs',activation = 'relu',hidden_layer_sizes = (100,100),random_state=0)
print("参数：solver='lbfgs',activation = 'relu',hidden_layer_sizes = (100,100),random_state=0")
mlp.fit(x_train,y_train)
mlp_score=mlp.score(x_test,y_test)
print('调参后回归模型得分',mlp_score)

#使用R2,MSE,MAE评测模型
y_predict=mlp.predict(x_test)
r2=r2_score(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
mae=mean_absolute_error(y_test,y_predict)
print("r2={}".format(r2))
print("mse={}".format(mse))
print("mae={}".format(mae))

