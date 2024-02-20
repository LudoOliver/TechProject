# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:14:52 2024

@author: xd21736
"""

#import torch as t
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp
import scipy.stats as stats
#%%
Size = np.linspace(1, 1e10,num=10000000)
#a = np.log(Size)
a = -0.05*((np.log(Size/7000))**2)
N_s = 550*np.exp(a)
plt.plot(Size,N_s,'k')
#plt.xlim((1), int(1e9))
#plt.ylim(1, 1000)
plt.title("Graph from paper")
plt.ylabel("$n_s$")
plt.xlabel("N")
#plt.figure()
#plt.hist(N_s)

#%%
n,x,_= plt.hist(N_s, bins=200, density=True) #was np.histo..
     #plt.title("Language distribution for")
density = stats.gaussian_kde(N_s)
plt.plot(x,density(x),color='red')

#%%


Sigma= 1/math.sqrt(2*math.pi)
Mu = 1e4*(Sigma**2)
MyApprox =[1*i for i in [Mu,Sigma]]

def LogNormal(DistributionParameters,x):
    Mu = DistributionParameters[0]
    Sigma=DistributionParameters[1]
    Coeff =1/(x*Sigma*math.sqrt(2*math.pi))
    Exponent = -1*(np.log(x)-Mu)**2/(2*(Sigma**2))
    return Coeff*np.exp(Exponent)

def FitToLogNormal(DistributionParameters):
    return [max(0.05,np.sum((LogNormal(DistributionParameters, Size)-N_s)**2))-0.05,LogNormal(DistributionParameters,Size)[10]-N_s[10]]
    #return [np.sum(LogNormal(DistributionParameters, Size)-N_s),np.linalg.norm(LogNormal(DistributionParameters, Size)-N_s)]
    
shape, loc, scale = sp.stats.lognorm.fit(N_s,scale=0.1) #,floc=0)#,fscale=1000,floc=-1e9)#(np.mean(N_s)/max(N_s)))
Graph = sp.stats.lognorm.pdf(Size ,shape, loc=loc, scale=scale)   
#plt.plot((Size), Graph, color='red', linewidth=2, label='Fitted Lognormal')
y=LogNormal(MyApprox,Size)
#plt.loglog(Size,y)
TestX = np.logspace(0,10)
#Attempt = LogNormal(10,0.1,TestX)
#%%
Solution = sp.optimize.root(FitToLogNormal,MyApprox)

Approx  = LogNormal(Solution.x, Size)

for i in np.logspace(-10,10,num=20):
    for j in np.logspace(-10, 10,num=20):
        MyApprox=[i,j]
        Solution = sp.optimize.root(FitToLogNormal,MyApprox)
        if Solution.success:
            print(f"Check mu{i},sigma{j}")


#print(Solution)
#plt.plot(TestX,Attempt)
#plt.figure()
#plt.loglog(Size,Approx)
#plt.figure()
#plt.loglog(Size,Approx)
# #%%
# shape, loc, scale = sp.stats.lognorm.fit(N_s,floc=0)#,fscale=1000,floc=-1e9)#(np.mean(N_s)/max(N_s)))
# #plt.figure()
# Graph = sp.stats.lognorm.pdf(Size ,shape, loc=loc, scale=scale)
# #Graph =Graph/max(Graph)*max(N_s)
# # plt.figure(figsize=(10, 5))
# # plt.hist(N_s, bins=50, density=True, edgecolor='black', alpha=0.6, label='Data')
# plt.figure()
# plt.plot(np.log(Size), Graph, color='red', linewidth=2, label='Fitted Lognormal')
# # plt.xlabel('Value')
# # plt.ylabel('Density')
# # plt.title('Fitted Lognormal Distribution')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
# #%%
# n,x = np.histogram(N_s, bins=20, density=True)
#     #plt.title("Language distribution for")
# density = stats.gaussian_kde(N_s)
# plt.loglog(Size,density(Size))#,label=f"T={0.3+0.1*i:.2f}",alpha=1-i*0.1)
# # #plt.hist(N_s,density=True)
# # plt.loglog(Size, Graph)
