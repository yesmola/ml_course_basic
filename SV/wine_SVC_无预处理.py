#!/usr/bin/env python
# coding: utf-8

# In[12]:


#导入SVM分类模型
from sklearn.svm import SVC
#导入拆分数据集工具
from sklearn.model_selection import train_test_split
#数据预处理
from sklearn.preprocessing import StandardScaler
#下载数据集
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score  

wine=load_wine()
x,y=wine.data,wine.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

svc=SVC() 
#使用默认参数
svc.fit(x_train,y_train)
svc_predict=svc.predict(x_test)
print("accuracy_score:{}".format(accuracy_score(y_test,svc_predict)))

