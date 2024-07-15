# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:27:02 2024

@author: xd21736
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
#import time
import cProfile
from scipy.stats import lognorm , norm, shapiro
from scipy import stats

import statsmodels.api as sm 
import pylab #as py 
#PrefTempValues = 
#PrefTempValues = [0.01,0.06,0.24,0.26,0.27,0.28,0.30]+[0.015*i for i in range(21,55)]
PrefTempValues = [0.015*i for i in range(1,55)]
PrefTempLabels = [f"{i:.2f}" for i in PrefTempValues]

#Pref

TempRange = PrefTempLabels
#TempRange = ThreshTempLabels

PrefBigArray = np.load("FPrefDistMatrix.npy")
PrefEnergyArray = np.load("FPrefEnergyMatrix.npy")
#%%
#%%
plt.figure()  
#for j in range(0,len(TempRange)):
for j in range(16,25):
    #if (j+1)%5==0:
    plt.figure()
    #plt.figure()
    OtherArray = PrefBigArray[j,:]     
    n,x = np.histogram(OtherArray, bins=20)#, density=True)
    density = stats.gaussian_kde(OtherArray)
    # if PrefTempLabels[j]=='0.57':
    #     plt.plot(x,density(x),label=f"T={TempRange[j]}",alpha=0.6,linewidth=1,color='b')
    plt.plot(x,(density(x)),label=f"T={TempRange[j]}",alpha=1,linewidth=1)
    #plt.fill_between(x,density(x),alpha=0.4)
   # plt.loglog(x,density(x),label=f"T={0.3+0.1*j:.2f}",alpha=1-j*0.1)
    #plt.plot(OtherArray)
   
    DistName = "PrefDistForLogLog.jpeg"
#plt.legend()
    plt.xticks([],[])
    plt.yticks([],[])
    #plt.title(f"Language Distribution of the Prefference Model at T=[{TempRange[j-5]},{TempRange[j]}]")
    plt.title(f"Language Distribution of the Prefference Model at T={TempRange[j]}")
    plt.xlabel("N speakers")
    plt.ylabel("$n_s$ number of languages")

#%%
plt.figure()  
for j in range(0,len(TempRange)):
    OtherArray = PrefBigArray[j,:]     
    n,x = np.histogram(OtherArray, bins=20, density=True)
        #plt.title("Language distribution for")
    density = stats.gaussian_kde(OtherArray)
    if PrefTempLabels[j]=='0.57':
        plt.plot(x,density(x),label=f"T={TempRange[j]}",alpha=0.6,linewidth=1,color='b')
    plt.plot(x,density(x),label=f"T={TempRange[j]}",alpha=0.6,linewidth=1)
    #plt.fill_between(x,density(x),alpha=0.4)
    #plt.loglog(x,density(x),label=f"T={0.3+0.1*j:.2f}",alpha=1-j*0.1)
   
DistName = "PrefDistForLogLog.jpeg"
#plt.legend()
plt.xticks([],[])
plt.yticks([],[])
plt.title("Language Distribution of the Prefference Model")
plt.xlabel("N speakers")
plt.ylabel("$n_s$ number of languages")
#plt.savefig(DistName,bbox_inches='tight', dpi=300)
# #%%
# plt.figure()
# TCritIndex = ThreshTempValues.index(0.57)
# CriticalArray = ThreshBigArray[TCritIndex,:]     
# #n,x = np.histogram(CriticalArray, bins=20, density=True)
# n,x,_ = plt.hist(CriticalArray, bins=20, density=True)
# density = stats.gaussian_kde(CriticalArray)
# y = density(x)
# plt.plot(x,y)

# #plt.xticks([],[])
# #plt.yticks([],[])
# plt.title(r"Language Distribution of the Threshold Model at $T_c$")
# plt.xlabel("N speakers")
# plt.ylabel("$n_s$ number of languages")
#%%
GraphTOCheck = 8
for j in range(GraphTOCheck,GraphTOCheck+1):
    OtherArray = np.log(PrefBigArray[j,:])     
    n,x = np.histogram(OtherArray, bins=20, density=True)
        #plt.title("Language distribution for")
    density = stats.gaussian_kde(OtherArray)
    if PrefTempLabels[j]=='0.57':
        plt.plot(x,density(x),label=f"T={TempRange[j]}",alpha=0.6,linewidth=1,color='b')
    plt.plot(x,density(x),label=f"T={TempRange[j]}",alpha=1,linewidth=0.5)
    plt.xticks([],[])
    plt.yticks([],[])
    plt.title(f"Language Distribution of the Prefference Model at T={TempRange[j]}")
    plt.xlabel("N speakers")
    plt.ylabel("$n_s$ number of languages")
#%%
plt.figure()
plt.cla()
EnergyVarVector= np.nanvar(PrefEnergyArray,axis=1) 

#for i in range(15):
   #plt.plot(PrefTempValues , PrefEnergyArray[:,i],alpha=0.3)
#     stats.probplot(ThreshBigArray[i,:], dist="norm", plot=pylab)
plt.plot(PrefTempValues ,EnergyVarVector)   
plt.xlabel("Temperature")
plt.ylabel("Energy")
#plt.yticks([])
#plt.xlim((0.4,0.7))#, kwargs)
#plt.ylim((-0.55,-0.2))
plt.title("Prefference Phase Behaviour")
# #%%%
# plt.figure()
#%%
"""Second Derrivative"""
plt.figure(figsize=(4,3),dpi=300)
x = np.array(PrefTempValues)
y = np.nanmean(PrefEnergyArray,axis=1)
ThreshEnergyArray = np.load("ThreshEnergyFinal.npy")
y = np.nanmean(ThreshEnergyArray,axis=1)
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

y_spl = UnivariateSpline(x,y,s=0,k=4)
y_spl_2d = y_spl.derivative(n=1)
x_range = np.linspace(x[0],x[-1],1000)
plt.plot(x_range,y_spl_2d(x_range))
plt.vlines(0.57, -2.4, 2.71,colors='r',linestyles='--',label="T=0.57",linewidth=1)#, kwargs)
plt.ylim((-2.3,2.7))#, kwargs)
plt.title("Critical Point of The Threshold Model")#, kwargs)
plt.xlabel("Temperature")
plt.ylabel(r"$\frac{dE}{dt}$   ",rotation=0,fontsize=12)
plt.legend()
# dy=np.diff(y,1)
# dx=np.diff(x,1)
# yfirst=dy/dx
# #And the corresponding values of x are :

# xfirst=0.5*(x[:-1]+x[1:])
# #For the second order, do the same process again :

# dyfirst=np.diff(yfirst,1)
# dxfirst=np.diff(xfirst,1)
# ysecond=dyfirst/dxfirst

# xsecond=0.5*(xfirst[:-1]+xfirst[1:])


#%%
#
# TCritIndex = ThreshTempValues.index(0.57)
# MeanVarBeforeCrit = np.mean(EnergyVarVector[:TCritIndex])
# MeanVarAfterCrit = np.mean(EnergyVarVector[TCritIndex:])
# BarNames= (r"Below $T_c$",r"Above $T_c$")
# plt.bar(BarNames, (MeanVarBeforeCrit,MeanVarAfterCrit))
# plt.ylabel("Variance in energy")
# plt.title("Variance above and below critical temperature")



#stats.probplot(ThreshBigArray, dist="norm", plot=pylab)