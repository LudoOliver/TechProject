# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:00:04 2024

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
plt.rcdefaults()

PrefBigArray = np.load("FPrefDistMatrix.npy")
PrefEnergyArray = np.load("FPrefEnergyMatrix.npy")
ThreshBigArray = np.load("ThreshBigArrayFinal.npy")
ThreshEnergyArray = np.load("ThreshEnergyFinal.npy")

ThreshClustersTc = np.load("ThreshClusters0.57.npz.npy")
ThreshClustersMin = np.load("ThreshClusters0.42.npz.npy")
PrefClustersTc = np.load("PrefClusters.npz.npy")

TempValues = [0.015*i for i in range(1,55)]
TempLabels = [f"{i:.2f}" for i in TempValues]

ThreshDistAtTc = ThreshBigArray[37,:] 
PrefDistAtTc = PrefBigArray[18,:]

RealWorldData = np.load("PopNumbers.npy").astype(int)
SortedReal = np.sort(RealWorldData)[::-1]

#ThreshClustersMin
SmallestDataSetSize = min(len(ThreshClustersMin),len(ThreshClustersTc),len(PrefClustersTc),len(RealWorldData))
def Normaliser(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

ThreshClustersTcShort = (np.sort(ThreshClustersMin)[::-1])[:SmallestDataSetSize]
ThreshClustersMinShort= (np.sort(ThreshClustersTc)[::-1])[:SmallestDataSetSize]
PrefClustersTcShort = (np.sort(PrefClustersTc)[::-1])[:SmallestDataSetSize]
#%%
ThreshTcAbove = ThreshClustersTc[np.where(ThreshClustersTc>1)]
plt.hist(np.log10(ThreshTcAbove))
#plt.xscale("log")
#plt.yscale("log")#
#%%
PrefDistAbove = PrefClustersTc[np.where(PrefClustersTc>1)]
plt.figure()
plt.xscale("log")#
plt.hist(np.log10(PrefDistAbove))
plt.yscale("log")
#%%
ThreshMinAbove = ThreshClustersMin[np.where(ThreshClustersMin>0)]
plt.figure()
plt.hist(np.log10(ThreshMinAbove),bins=20)
plt.xscale("log")#
plt.yscale("log")
#%%
plt.figure()
n,x,_ = plt.hist((SortedReal),bins=1000,density=True,alpha=0.8)

density = stats.gaussian_kde((SortedReal))
y = density(x)
plt.yscale("log")
plt.figure()
plt.plot(x,y,linewidth=4)
plt.xlim(1,3e8)
plt.yscale("log")#, kwargs)
#plt.xscale("log")#, kwargs)

#%%
ClusterDict = {"Thresh at 0.42":ThreshClustersTcShort,
               "Thresh at 0.57":ThreshClustersMinShort,
               "Pref at 0.28":PrefClustersTcShort ,
               "Real":SortedReal}
AboveDict = {"ThreshTCAbove":ThreshTcAbove,
             "PrefDistABove"}



end= SmallestDataSetSize
n = 100
base = (np.power(end,1/n))
BinBounds = np.unique([math.floor(base**i) for i in range(0,n)])

for i,j in ClusterDict.items():#, kwargs):
    try:
        OtherArray = Normaliser(j)
        
        #OtherArray = (((j-np.mean(j))/np.std(j)))
        plt.hist(OtherArray,label=i,bins=n,density=True,alpha=0.8)
        plt.yscale("log")
        n,x = np.histogram(OtherArray, bins=30, density=True)
        #plt.title("Language distribution for")
    #plt.scatter(n[10:],x[10:-1])
        #density = stats.gaussian_kde(OtherArray)
    
        #y = density(x)
        #if i=="Real":
          #  x = x[np.where(y>0)]
           # print(len(x))
            #y = y[np.where(y>0)]
            #print(y)
    #x = x[np.where(y>1e-1)]
    #y= density(Normaliser(x))
    #print(i,np.sum(y*x))
        #plt.loglog(((x)),(y),label=i)#density(x))
    except:
        print(f"Couldnt do {i}")
    plt.title("Comparing Language Distributions")
    plt.xticks([],[])
    plt.yticks([],[])
    plt.title("Language Distribution of Each Model")
    plt.xlabel("N speakers")
    plt.ylabel("$n_s$ number of languages")
    plt.legend()
    #plt.figure()

#%%
plt.figure()
end= SmallestDataSetSize
n = 1000
base = (np.power(end,1/n))
BinBounds = np.unique([math.floor(base**i) for i in range(0,n)])
for i,j in ClusterDict.items():
    NormZarray = ((j-np.mean(j))/np.std(j))
    #plt.hist(NormZarray,bins=20,density=True)
    n,x = np.histogram(NormZarray, bins=BinBounds)# density=True)
        #plt.title("Language distribution for")
    #plt.scatter(n[10:],x[10:-1])
    density = stats.gaussian_kde(NormZarray)
    
    y = density(x)
    x = x[np.where(y>0)]
            #print(len(x))
    y = y[np.where(y>0)]
    plt.loglog(x,y,label=i)
    plt.legend()
    #plt.figure()
    #plt.xlim((1,20))
           # print(y)
    #x = x[np.where(y>1e-1)]
    #y= density(Normaliser(x))
    #print(i,np.sum(y*x))
        #plt.loglog(((x)),(y),label=i)#density(x))
#%%
plt.figure()
end= SmallestDataSetSize
n = 10
base = (np.power(end,1/n))
BinBounds = np.unique([math.floor(base**i) for i in range(0,n)])
x=[]
y=[]
for i in range(len(BinBounds)):
    TotalAtBin = np.sum((ClusterDict["Pref at 0.28"])[BinBounds[i-1]:BinBounds[i]])
    x.append(np.log10(i))
    y.append(np.log10(TotalAtBin))
    
plt.plot(x,y)