#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# In[2]:


X_traindr=np.loadtxt('train_X.csv',delimiter=',').T
Y_traindr=np.loadtxt('train_label.csv',delimiter=',').T
X_testdr=np.loadtxt('test_X.csv',delimiter=',').T
Y_testdr=np.loadtxt('test_label.csv',delimiter=',').T


# In[3]:


type(X_traindr)


# In[4]:


Y_traindr


# In[5]:


Y_traindr.shape


# In[6]:


X_traindr.shape


# # VIEWING OF IMAGE
# 
# 

# In[7]:


index=random.randrange(0,X_traindr.shape[1])
plt.imshow(X_traindr[:, index].reshape(28,28), cmap='gray')
plt.show()


# In[8]:


def tanh(x):
  return np.tanh(x)

def relu(x):
  return np.maximum(x,0)

def softmax(x):
  expX=np.exp(x)
  return expX/np.sum(expX ,axis=0)


# ### DERIVATIVE

# In[9]:


def derivative_tanh(x):
  return (1-np.power(x,2))

def derivative_relu(x):
  return np.array(x>0,dtype=np.float32)


# ### INITIAL PARAMETER

# In[10]:


def initial_parameter(n_x,n_h,n_y):
  w1=np.random.randn(n_h,n_x)*0.001
  b1=np.zeros((n_h,1))
  w2=np.random.randn(n_y,n_h)*0.001
  b2=np.zeros((n_y,1))
  #we made parameter dict to return whole dict
  parameters={
      "w1":w1,
      "b1":b1,
      "w2":w2,
      "b2":b2
  }
  return parameters


# ### forward propagation
# 
# 

# In[11]:


def forward_propagation(x,parameters):
  w1=parameters['w1']
  b1=parameters['b1']
  w2=parameters['w2']
  b2=parameters['b2']

  z1=np.dot(w1,x)+ b1
  a1=relu(z1)
  z2=np.dot(w2,a1)+ b2
  a2=softmax(z2)

  forward_prop = {
    "z1":z1,
    "a1":a1,
    "z2":z2,
    "a2":a2
  }

  return forward_prop


# COST FUNCTION

# In[12]:


def cost_fxn(a2,y):
  m = y.shape[1]
  cost=-(1/m)*np.sum(y*np.log(a2))

  return cost


# ### back propagation

# In[13]:


def back_prop(x,y,parameters,forward_prop):
  w1=parameters["w1"]
  b1=parameters["b1"]
  w2=parameters["w2"]
  b2=parameters["b2"]
  z1=forward_prop["z1"]
  a1=forward_prop["a1"]
  z2=forward_prop["z2"]
  a2=forward_prop["a2"]

  m=x.shape[1]
  dz2=(a2-y)
  dw2=(1/m)*np.dot(dz2,a1.T)
  db2=(1/m)*np.sum(dz2,axis = 1,keepdims= True)
  dz1=(1/m)*np.dot(w2.T,dz2)*derivative_relu(a1)
  dw1=(1/m)*np.dot(dz1,x.T)
  db1=(1/m)*np.sum(dz1,axis=1, keepdims=True)

  gradients={
      "dw1":dw1,
      "db1":db1,
      "dz1":dz1,
      "dw2":dw2,
      "db2":db2,
      "dz2":dz2
  }
  return gradients


# In[14]:


def update_parameter(parameters,gradients,learning_rate):
  w1=parameters['w1']
  b1=parameters['b1']
  w2=parameters['w2']
  b2=parameters['b2']

  dw1=gradients['dw1']
  db1=gradients['db1']
  dw2=gradients['dw2']
  db2=gradients['db2']

  w1=w1-learning_rate*dw1
  b1=b1-learning_rate*db1
  w2=w2-learning_rate*dw2
  b2=b2-learning_rate*db2

  parameters={
      "w1":w1,
      "b1":b1,
      "w2":w2,
      "b2":b2
  }
  return parameters


# In[15]:


def model(x,y,n_h,learning_rate,iterations):
  n_x=x.shape[0]
  n_y=y.shape[0]
  cost_list=[]
  parameter=initial_parameter(n_x,n_h,n_y)
  for i in range(iterations):
    forward_prop=forward_propagation(x,parameter)

    cost=cost_fxn(forward_prop['a2'],y)

    gradients=back_prop(x,y,parameter,forward_prop)

    parameter=update_parameter(parameter,gradients,learning_rate)
    
    cost_list.append(cost)
    if(i%(iterations/10)==0):
      print("cost after",i,"iteration is", cost)
  return parameter,cost_list


# In[16]:


iteration=100
n_h=1000
learning_rate=0.1
parameters,cost_list=model(X_traindr,Y_traindr,n_h,learning_rate,iteration)


# In[17]:


x=np.arange(0,iteration)
plt.plot(x,cost_list);


# FOR ACCURACY CHECK
# 

# In[18]:


def accuracy(inp,labels,parameters):
    
    forwardprop=forward_propagation(inp,parameters)
    a_out=forwardprop["a2"]
    a_out=np.argmax(a_out,axis=0)

    y_out=np.argmax(labels,0)
    a_out==y_out
    accuracy=np.mean(a_out==y_out)*100
    return accuracy 


# In[19]:


print("accuracy of training",accuracy(X_traindr,Y_traindr,parameters),"%")
print("accuracy of testing",accuracy(X_testdr,Y_testdr,parameters),"%")


# In[20]:


X_testdr.shape


# # PREDICTION

# In[21]:


idx=random.randrange(0,X_testdr.shape[1])
plt.imshow(X_testdr[:, idx].reshape(28,28), cmap='gray')
plt.show()
forwardprop=forward_propagation(X_testdr[:,idx].reshape(X_testdr.shape[0],1),parameters)
a_out=forwardprop["a2"]
a_out=np.argmax(a_out,axis=0)
print("this is model prediction",a_out[0])


# In[ ]:





# In[ ]:




