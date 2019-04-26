#!/usr/bin/env python
# coding: utf-8

# In[9]:


#导入Logistic回归模型
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.metrics import classification_report

#1.数据准备
news = fetch_20newsgroups_vectorized()
data=news.data
target=news.target
#2.将数据拆分为训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=1)
#3.初始化LR函数
lr=LogisticRegression()
#4.使用LR拟合训练集，得到预测模型
predict_model=lr.fit(x_train,y_train)

#从预测模型中获得直线系数和截距
coef=predict_model.coef_            #直线系数
intercept=predict_model.intercept_  #截距
print("coef={}".format(coef))
print("intercept={}".format(intercept))
#分类结束，评测模型:自带准确率评分以及精确度，召回率和F1值
y_predict=predict_model.predict(x_test)
print("模型的准确率为:{}".format(predict_model.score(x_train,y_train)))
print(classification_report(y_test, y_predict))
print("test={}".format(y_test))
print("predict={}".format(y_predict))


# In[ ]:




