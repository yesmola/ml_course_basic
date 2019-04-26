#!/usr/bin/env python
# coding: utf-8

# In[13]:


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score  
import numpy as np

wine=load_wine()
x=wine.data[:,:2]
y=wine.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#多层感知机回归模型
mlp=MLPClassifier(random_state=0)
mlp.fit(x_train,y_train)
mlp_predict=mlp.predict(x_test)
print("默认参数:accuracy_score:{}".format(accuracy_score(y_test,mlp_predict)))


# In[36]:


#mlp=MLPClassifier(solver='lbfgs',random_state=0)
#mlp=MLPClassifier(solver='lbfgs',activation = 'logistic',random_state=0)
#mlp=MLPClassifier(solver='lbfgs',activation = 'logistic',random_state=0)
mlp=MLPClassifier(solver='lbfgs',activation = 'logistic',hidden_layer_sizes = (100,100,100),alpha = 1e-5,random_state=0)
print("参数：solver='lbfgs',activation = 'logistic',hidden_layer_sizes = (100,100,100),alpha = 1e-5,random_state=0")
mlp.fit(x_train,y_train)
mlp_predict=mlp.predict(x_test)
print("调整参数后:accuracy_score:{}".format(accuracy_score(y_test,mlp_predict)))

