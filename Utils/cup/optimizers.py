#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cupy as np
def params():
    print("RMS_prop(),Adam(theta),Momentum(theta)")
    
class RMS_prop:
    def __init__(self):
        self.lr=0.1
        self.beta=0.9
        self.epislon=1e-8
        self.s=0
        self.t=0
        
    def initial(self):
        self.s = 0
        self.t = 0
    
    def New_theta(self,theta,gradient,eta):
        self.lr = eta
        self.t += 1
        g=gradient
        self.decay=1e-4
        self.s = self.beta*self.s + (1-self.beta)*(g*g)
        theta -= self.lr*((g/pow(self.s+self.epislon,0.5))+self.decay*theta)
        return theta

class Adam:
    def __init__(self,theta):
        self.lr=0.01
        self.beta1=0.9
        self.beta2=0.999
        self.epislon=1e-8
        self.m=[np.zeros(ms.shape) for ms in theta]
        self.s=[np.zeros(ms.shape) for ms in theta]
        self.t=0
    
    def New_theta(self,theta,gradient,eta):
        self.t += 1
        if type(eta) == list:
            self.lr = eta*1
        if type(eta) == float:
            self.lr=[eta*np.ones((theta_s.shape)) for theta_s in theta]
        self.decay=1e-4
        g=gradient*1
        theta2 = [np.zeros(ms.shape) for ms in theta]
        for l in range(len(gradient)):
            self.m[l] = self.beta1*self.m[l] + (1-self.beta1)*g[l]
            self.s[l] = self.beta2*self.s[l] + (1-self.beta2)*(g[l]*g[l])
            self.mhat = self.m[l]/(1-self.beta1**self.t)
            self.shat = self.s[l]/(1-self.beta2**self.t)
            theta2[l] = theta[l]-self.lr[l]*((self.mhat/(pow(self.shat,0.5)+self.epislon))+self.decay*theta[l])
        return theta2*1
class Momentum:
    def __init__(self,theta):
        self.lr=0.01
        self.epislon=1e-8
        self.velocity=(np.zeros(shape = theta.shape))
        self.theta2 = (np.zeros(shape = theta.shape))
    def New_theta(self,theta,gradient,lr):
        self.lr = lr*1
        eta = 0.9
        g=gradient*1
        self.velocity = self.velocity*eta+lr*g
        self.theta2 = theta - self.velocity
        return self.theta2
class BN_layer(object):
    def __init__(self,channel,shape=2):
        self.shape=shape
        if shape == 2:
            self.axis = (0,1,2)
            self.gamma = np.random.normal(0,0.01,(1,1,1,channel))
            self.beta =  np.random.normal(0,0.01,(1,1,1,channel))
            self.running_mean = np.zeros((1,1,1,channel))
            self.running_var = np.zeros((1,1,1,channel))
        if shape == 1:
            self.axis = (1)
            self.gamma = np.random.normal(0,0.01,(channel,1))
            self.beta =  np.random.normal(0,0.01,(channel,1))
            self.running_mean = np.zeros((channel,1))
            self.running_var = np.zeros((channel,1))
        self.momentum = 0.9
        self.beta_Mo =  Momentum(self.beta)
        self.gamma_Mo =   Momentum(self.gamma)
    def batch_norm(self,x,mean, var, gamma, beta, eps=1e-5,grad=False):
        x_hat = (x - mean) / np.sqrt(var + eps)
        output = gamma*x_hat + beta
        if (grad==True):
            variable = [(x, x_hat, mean, var, eps), (gamma, None), (beta, None)]
            return output,variable
        else:
            return output
    def BN_backward(self,x,dout,place,train=False):
        batch_mean = (np.mean(x, axis=self.axis, keepdims=True))#1111的尺寸
        batch_var = (np.var(x, axis=self.axis, keepdims=True))#1111
        self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * batch_mean
        self.running_var = (1. - self.momentum) * self.running_var + self.momentum * batch_var
        if (train == False):
            return self.batch_norm(x, self.running_mean, self.running_var, self.gamma, self.beta, eps=1e-5,grad=False)

        if (train == True):
            output,variable = self.batch_norm(x, batch_mean, batch_var, self.gamma, self.beta, eps=1e-5,grad=True)
            if place == 0:
                return output*1
            else:
                x, x_hat, mean, var, eps = variable[0]
                grad_x_hat = dout * self.gamma
                self.beta = self.beta_Mo.New_theta(self.beta,np.sum(dout, axis=self.axis,keepdims=True),0.01)
                self.gamma = self.gamma_Mo.New_theta(self.gamma,np.sum(x_hat * dout, axis=self.axis,keepdims=True),0.01)
                n = dout.size / dout.shape[self.shape-3]
                dx = n * grad_x_hat - np.sum(grad_x_hat, axis=self.axis,keepdims=True) - x_hat * np.sum(x_hat * grad_x_hat, axis=self.axis, keepdims=True)
                dx = dx / (n * np.sqrt(var + eps))
                return dx


            

        

        
        
