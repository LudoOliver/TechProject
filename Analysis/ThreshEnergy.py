# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:48:37 2024

@author: farka
"""

#import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import math



from matplotlib.collections import PolyCollection
#mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "k", "c"]) 
N=12
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0.1,0.8,N)))

#import psutil
#import tracemalloc
#tracemalloc.start()
#BluePebResult = np.load("BluePebbleOut.npz")
#prin
#SimulatedArray = BluePebResult['arr_0']
#TestingPassBack = BluePebResult['arr_1']
#[GridSize,Temp,NTimeSteps,NLangFeatures] = [i for i in TestingPassBack]

def LanguageDist(MatrixForAnalysis):
    BigList=[]
    for i in MatrixForAnalysis:
        BigList.append(np.array2string(i))
    SpinDict = {}
    for j in BigList:
        if j in SpinDict:
            SpinDict[j] += 1
        else:
            SpinDict.update({j: 1})
    SpinDict = dict(sorted(SpinDict.items(),key=lambda item: item[1],reverse=True))   
    return SpinDict

def LanguageVector(MatrixForAnalysis):
    PossibleLanguages = 2**(np.shape(MatrixForAnalysis)[1])
    
    BigList=[]
    for i in MatrixForAnalysis:
        BigList.append(np.array2string(i))
    SpinDict = {}
    for j in BigList:
        if j in SpinDict:
            SpinDict[j] += 1
        else:
            SpinDict.update({j: 1})
    SpinDict = dict(sorted(SpinDict.items(),key=lambda item: item[1],reverse=True))
    SpinArray = np.array(list(SpinDict.values()))
    if len(SpinArray) < PossibleLanguages:
        #print("eek")
        #print("possible")
        SpinArray = np.pad(SpinArray, (0,PossibleLanguages-len(SpinArray)), 'constant')
        #print(len(SpinArray))
    return SpinArray

def ResultsFor(mode,temperature,length):
    FileName = f"{mode}L{length}T{temperature:.2f}.npz"
    DataInFile = np.load(FileName)
    return DataInFile



BigArray = np.zeros([10,256])
plt.figure()   
for i in range(6,16):
    #plt.figure()#,figsize=(10,10))
    
    BluePebbleResult = ResultsFor("Thresh", 0.05*i, 300)
    OtherArray =np.zeros(256)
    for j in BluePebbleResult.files:
    
        Result = LanguageVector(BluePebbleResult[j])
        OtherArray += Result
    #     plt.plot(Result,label=f"Attempt{j}")
    BigArray[i-6,:] = OtherArray
    n,x = np.histogram(OtherArray, bins=20, density=True)
    #plt.title("Language distribution for")
    density = stats.gaussian_kde(OtherArray)
    plt.plot(x,density(x),label=f"T={0.05*i:.2f}",alpha=0.9)
    #plt.loglog(x,density(x),label=f"T={0.3+0.1*i:.2f}",alpha=1-i*0.1)
    
DistName = "ThreshDistForVaryingT.jpeg"
plt.legend()
plt.xticks([],[])
plt.yticks([],[])
plt.title("Language Distribution of the Threshold Model")
plt.xlabel("N speakers")
plt.ylabel("$n_s$ number of languages")
#plt.savefig(DistName,bbox_inches='tight', dpi=300)
  
#%%
plt.figure()  
for j in range(6,16):
    OtherArray = BigArray[j,:]     
    n,x = np.histogram(OtherArray, bins=20, density=True)
        #plt.title("Language distribution for")
    density = stats.gaussian_kde(OtherArray)
    plt.plot(x,density(x),label=f"T={0.05*j:.2f}",alpha=0.9)
    plt.fill_between(x,density(x),alpha=0.4)
    #plt.loglog(x,density(x),label=f"T={0.3+0.1*j:.2f}",alpha=1-j*0.1)
   
DistName = "ThreshDistForLogLog.jpeg"
plt.legend()
plt.xticks([],[])
plt.yticks([],[])
plt.title("Language Distribution of the Threshold Model")
plt.xlabel("N speakers")
plt.ylabel("$n_s$ number of languages")
#plt.savefig(DistName,bbox_inches='tight', dpi=300)
