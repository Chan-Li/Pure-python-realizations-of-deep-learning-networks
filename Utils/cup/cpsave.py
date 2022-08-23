#!/usr/bin/env python
# coding: utf-8

# In[158]:


import numpy as np
import cupy as cp
def para():
    print("class name: cp_save(path)")
class cp_save(object):
    def __init__(self,path):
        self.path=path
    def para(self):
        print("main function: cp_s(cp_arr)")
    def cp_s(self,cp_arr):
        self.data=cp_arr
        accx =[]
        accy =[]
        for i in range(len(self.data)):
            accx.append(cp.asarray(self.data[i])*1)
            
        for i in range(len(self.data)):
            accy.append(cp.asnumpy(accx[i])*1)
        np.save(self.path,(accy),allow_pickle=True)
     

