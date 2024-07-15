# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:20:35 2024

@author: xd21736
"""

#import torch as t
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp
import scipy.stats as stats
from scipy.stats import shapiro
#%%
Size = np.linspace(1, 1e10,num=10000)
N = np.linspace(1, 1e10,num=500)
NLog = np.logspace(1, 10,num=1000)
#a = np.log(Size)
a = -0.05*((np.log(Size/7000))**2)
N_s = 550*np.exp(a)

a = -0.05*((np.log(NLog/7000))**2)
LogN_s = 550*np.exp(a)


def LogNormal(DistributionParameters,x):
    Mu = DistributionParameters[0]
    Sigma=DistributionParameters[1]
    Coeff =1/(x*Sigma*math.sqrt(2*math.pi))
    #print(Coeff)
    Exponent = -1*((np.log(x)-Mu)**2)/(2*(Sigma**2))
    return Coeff*np.exp(Exponent)
mu2 = 0.5
sigma2 = 1
def Normal(x):
    return np.exp(-1*np.square((x-mu2)/sigma2))*0.5
Scale =30
Mu = 0.5
Sigma = 1

x=np.linspace(0,10,num=2000)
x2= np.linspace(-4,6)
#x=np.logspace(np.log10(10),np.log10(400),num=2000)
y=LogNormal([Mu,Sigma],x)
#plt.xlim((-4,4))
#MuEstimate = (1/len(y))*np.sum(np.log(y)) # I dont understand
#SigmaEstimate = np.sqrt((1/len(y))*np.sum(np.square(np.log(y)-MuEstimate)))

#EstimateY = LogNormal([MuEstimate,SigmaEstimate],x)
plt.figure(figsize=(4.5,3),dpi=300)
plt.plot(x,y,lw=4,label="Log-Normal")
plt.xticks([],[])
plt.yticks([],[])
plt.plot(x2+4,Normal(x2),lw=4,linestyle = "--",label="Normal")
plt.legend()
#plt.loglog(x,y)
#plt.hist(y)
#plt.plot(x,np.log(EstimateY))

#%%
plt.figure()
plt.hist(N_s)
plt.figure()
plt.hist(LogN_s)
plt.figure()
plt.hist(y) # Plots the histograms of our daTA - SEEMS that not using logspace is bad
plt.figure()

LnOfN_s = np.log(N_s)
plt.hist(LnOfN_s)
plt.title('Natural Log Transformed Close Values in a Histogram')

#stat, p = shapiro(LnOfN_s)


