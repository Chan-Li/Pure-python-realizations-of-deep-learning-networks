#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cupy as np
def params():
    print("The functions:relu(y),drelu(y),softmax(y),divi_(lr0,global_step,decay_step),mini_batch_generate(mini_batch_size,data1,label1), sigmoid(beta,mat),dsigmoid(y),turn_2_zero(x)")
def relu(y):
    tmp = y.copy()
    tmp[tmp < 0] = 0
    return tmp
def drelu(x):
    tmp = x.copy()
    tmp[tmp >= 0] = 1
    tmp[tmp < 0] = 0
    return tmp

def softmax(y):
    y = y - np.array(y.max(axis=0),ndmin=2)
    exp_y = np.exp(y) 
    sumofexp = np.array(exp_y.sum(axis=0),ndmin=2)
    softmax = exp_y/sumofexp
    return softmax

def divi_(lr0,global_step,decay_step):
    return lr0*(0.5**((int(global_step/decay_step))))

def uni_permu(a,b,direction):
    if direction ==1:
        p = np.random.permutation(len(a.T))
        return np.array((a.T[p]).T), np.array((b.T[p]).T)
    if direction == 0:
        p = np.random.permutation(len(a))
        return np.array((a[p])), np.array((b[p]))
def mini_batch_generate2(mini_batch_size,data1,label1):
    data = np.array(data1*1)
    label = np.array(label1*1)
    if (data.shape[1]%mini_batch_size == 0):
        n=data.shape[1]
    else:
        n = (int(data.shape[1]/mini_batch_size))*mini_batch_size
    data,label = uni_permu(data,label,1)
    mini_batches = np.array([data[:,k:k+mini_batch_size] for k in range(0,n,mini_batch_size)])
    mini_batches_labels =np.array([label[:,k:k+mini_batch_size] for k in range(0,n,mini_batch_size)])
    return (mini_batches),(mini_batches_labels)
def mini_batch_generate3(mini_batch_size,data1,label1):
    data = np.array(data1*1)
    label =np.array(label1*1)
    if (data.shape[0]%mini_batch_size == 0):
        n=data.shape[0]
    else:
        n = (int(data.shape[0]/mini_batch_size))*mini_batch_size
    data,label = uni_permu(data,label,0)
    mini_batches = np.array([data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)])
    mini_batches_labels =np.array([label[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)])
    mini_batches = mini_batches.reshape(int(data.shape[0]/mini_batch_size),mini_batch_size,data1.shape[1],data1.shape[2],data1.shape[3])
    mini_batches_labels = mini_batches_labels.reshape(int(label.shape[0]/mini_batch_size),mini_batch_size,label.shape[1],1)
    return (mini_batches),(mini_batches_labels)
def sigmoid(mat,beta=1.0):
    return (1.0/(1+pow(np.e,-beta*mat))).reshape(mat.shape)
def dsigmoid(mat,beta=1.0):
    return (beta*sigmoid(mat,beta)*(1-sigmoid(mat,beta)))
def turn_2_zero(x):
    return np.int64(x>0)
def scale(x1,alpha1,alpha2):
    x = x1*1
    x[x1<alpha1] =alpha2
    return np.array(x)
def tanh(x):
    return np.tanh(x)
def dtanh(x):
    return (1-(np.tanh(x)**2))
