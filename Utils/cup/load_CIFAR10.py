from PIL import Image
import os
import pickle
import cupy as cp
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float")
        Y =cp.array(Y)
        return X, Y
def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y) 
    xs = cp.array(xs)
    ys = cp.array(ys)
    Xtr = cp.concatenate(xs)#使变成行向量
    Ytr = cp.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte
def _one_hot(labels, num):
    size= labels.shape[0]
    label_one_hot = cp.zeros([size, num])
    for i in range(size):
        label_one_hot[i, cp.squeeze(labels[i])] = 1
    return label_one_hot
def data_generate(root):
    Xtr_ori, Ytr_ori, Xte_ori, Yte_ori=load_CIFAR10(root)
    Xtr = (Xtr_ori.astype('float32') / 255.0)
    Xte = (Xte_ori.astype('float32') / 255.0)
    Ytr=_one_hot(Ytr_ori, 10).reshape(len(Ytr_ori),10,1)
    Yte=_one_hot(Yte_ori, 10).reshape(len(Yte_ori),10,1)
    del Xtr_ori, Ytr_ori, Xte_ori, Yte_ori
    return Xtr,Ytr,Xte,Yte
# 去中心化
# def data_generate(root):
#     Xtr_ori, Ytr_ori, Xte_ori, Yte_ori=load_CIFAR10(root)
#     Xtr = cp.array(Xtr_ori.astype('float32'))
#     Xte = cp.array(Xte_ori.astype('float32'))
#     Ytr=_one_hot(Ytr_ori, 10).reshape(len(Ytr_ori),10,1)
#     Yte=_one_hot(Yte_ori, 10).reshape(len(Yte_ori),10,1)
#     del Xtr_ori, Ytr_ori, Xte_ori, Yte_ori
#     mean   = [125.307, 122.95, 113.865]
#     std   = [62.9932, 62.0887, 66.7048]
#     mean2 = cp.array([126.02, 123.71, 114.85])
#     std2 = cp.array([62.8964, 61.9375, 66.7061])
#     for i in range(3):
#         Xtr[:,:,:,i] = (Xtr[:,:,:,i] - mean[i]) / std[i]
#         Xte[:,:,:,i] = (Xte[:,:,:,i] - mean2[i]) / std2[i]  
#     return Xtr,Ytr,Xte,Yte

def turn_2_zero(x1):
    import numpy
    x = cp.asnumpy(x1)
    y = numpy.int64(x>0)
    return cp.asarray(y)
def flip_ran(data1,p,dim):
    data = data1*1
    ran = turn_2_zero(p*cp.ones((data.shape))-cp.array([cp.random.uniform(0,1)*cp.ones((xx.shape)) for xx in data] ))
    data_b = ran*cp.flip(data*1,dim)+(1-ran)*data*1
    return data_b
def light_adjust(data1, a, b,p):
    data = data1*1
    temp = data*a+b
    temp2 = temp*1
    temp2[temp>1] = 1.0
    ran = turn_2_zero(p*cp.ones((data.shape))-cp.array([cp.random.uniform(0,1)*cp.ones((xx.shape)) for xx in data] ))
    data_b = ran*temp2+(1-ran)*data*1
    return data_b
def rotate(data1,p):
    data = data1*1
    temp2 = (cp.rot90(data, 1,axes=(1, 2)))
    ran = turn_2_zero(p*cp.ones((data.shape))-cp.array([cp.random.uniform(0,1)*cp.ones((xx.shape)) for xx in data] ))
    data_b = ran*temp2+(1-ran)*data*1
    return data_b
def Augmentation(data1,label1):
    num = 4
    data=cp.array(data1*1)
    label=cp.array(label1*1)
    lenth = data.shape[0]
    datab = cp.zeros((lenth*num,data.shape[1],data.shape[2],data.shape[3]))
    labelb = cp.zeros((lenth*num,label.shape[1],label.shape[2]))
    datab[0:lenth] = data*1
    datab[lenth:2*lenth] = flip_ran(data,0.7,2)
    datab[2*lenth:3*lenth] = rotate(datab[lenth:2*lenth],0.7)
    datab[3*lenth:4*lenth] = light_adjust(datab[2*lenth:3*lenth], cp.random.uniform(0.2,3.0), cp.random.uniform(0.0,1.5),0.7)
    labelb[0:lenth] = label*1
    labelb[1*lenth:2*lenth] = label*1
    labelb[2*lenth:3*lenth] = label*1
    labelb[3*lenth:4*lenth] = label*1
    del data1,label1,data,label
    return datab,labelb
