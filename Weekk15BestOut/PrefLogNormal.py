# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:33:39 2024

@author: farka
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp
from scipy.stats import lognorm , norm, shapiro
import statsmodels.api as sm
import pylab
import TraditionalThreshEnergy
def ResultsFor(mode,temperature,length):
    FileName = f"{mode}L{length}T{temperature:.2f}.npz"
    DataInFile = np.load(FileName)
    return DataInFile
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


TempVec = [0.015*i for i in range(20,40)]+[0.1*i for i in range(3,8)]
TempRange = [f"{i:.2f}" for i in TempVec]
PrefBigArray = np.zeros([len(TempRange),256])
plt.figure()   
for i in range(0,len(TempRange)):
    BluePebbleResult = ResultsFor("Pref", float(TempRange[i]), 300)
    OtherArray =np.zeros(256)
    for j in BluePebbleResult.files:
    
        Result = LanguageVector(BluePebbleResult[j])
        OtherArray += Result
    PrefBigArray[i,:] = OtherArray
#%%
plt.plot(TempRange,np.mean(PrefBigArray,axis=1))
plt.figure()
#plt.plot(TempRange,np.mean(BigArray,axis=0))
plt.figure()
for i in range(len(TempRange)):
    plt.hist(PrefBigArray[i,:],alpha=0.1)
#a,b=shapiro([1,23,4])
#print(a,b)
#%%
TestStatArray = np.zeros(len(TempRange))
TestProbArray = np.zeros(len(TempRange))
LogTestStatArray = np.zeros(len(TempRange))
LogTestProbArray = np.zeros(len(TempRange)) #Work out if need indexing
for i in range(0,len(TempRange)):
    TestStatArray[i],TestProbArray[i] = shapiro(PrefBigArray[i,:])
    LogTestStatArray[i],LogTestProbArray[i] = shapiro(np.log(PrefBigArray[i,:]))  
#plt.locator_params(axis='x',nbins=10)    
plt.scatter(np.array(TempRange),TestStatArray,label="Normal")
plt.scatter(np.array(TempRange),LogTestStatArray,label="Lognormal")
plt.xlabel("Temperature")
plt.xticks(ticks=np.array(TempRange)[::3])
plt.ylabel("Shapiro-Wilk Statistic")
plt.legend()
plt.title("Preference Distribution Statistics")
plt.figure()
#plt.locator_params(axis='x',nbins=10)    
plt.scatter(np.array(TempRange),TestProbArray,label="Normal")
plt.scatter(np.array(TempRange),LogTestProbArray,label="Lognormal")
plt.xticks(ticks=np.array(TempRange)[::3])
plt.xlim([10,25])
plt.hlines(0.05,xmin=0,xmax=34 ,color='r',linewidth=0.5,linestyles='--')
#plt.hlines(0.5, color='r',linewidth=0.5)
plt.xlabel("Temperature")
plt.ylabel("Shapiro-Wilk P value")
plt.legend()
plt.title("Threshold Distribution P-Value")
#%%
TempRange = [0.015*i for i in range(20,40)]+[0.1*i for i in range(3,8)]
TempRange.sort()

# def LatticeGenerate(SinglePopMatrix,GridSize=300):
    
#     Lattice = []
#     for i in range(0,GridSize**2):
#         Lattice.append(Speaker(i, SinglePopMatrix[i,:]))
#     return Lattice


EnergyMatrix = np.zeros([len(TempRange),15])+np.nan
for i in range(0,len(TempRange)):
    Data = ResultsFor('Pref', float(TempRange[i]), 300)
    for j in range(0,len(Data.files)):
        Population = TraditionalThreshEnergy.LatticeGenerate(Data[str(j)])
        TraditionalThreshEnergy.SpinVisualiser()
        EnergyMatrix[i,j] = TraditionalThreshEnergy.FastEnergy()
        #print(f"{i*10+j}% done")
plt.cla()
EnergyVector= np.nanvar(EnergyMatrix,axis=1) 
plt.plot([float(i) for i in TempRange],EnergyVector,color='b')   
plt.xlabel("Temperature")
plt.ylabel("Variance")
plt.yticks([])
#plt.xlim((0.4,0.7))#, kwargs)
#plt.ylim((-0.55,-0.2))
plt.title("Preference Phase Behaviour")
#%%
test = np.random.normal(0,1, 1000)

#sm.qqplot(test, line='45')
#pylab.show()

LogData = np.random.lognormal(mean=1,sigma=1,size=1000)
LogMu,LogSigma,LogLoc = lognorm.fit(LogData)
NormData = np.random.normal(loc=1,scale=1,size=1000)
#sm.qqplot(NormData,line='45')
#sm.qqplot(LogData,line='45')
#sm.qqplot(np.log(LogData),line='45')
NLog = np.logspace(1, 10,num=100)
Size = np.linspace(1, 1e10,num=1000)
a = -0.05*((np.log(NLog/7000))**2)
b = -0.05*((np.log(Size/7000))**2)
N_s = 550*np.exp(a)
LinN_s= 550*np.exp(b)
a1 = shapiro(NormData)
a2 = shapiro(LogData)
a3 = shapiro(np.log(LogData))
a4 = shapiro(N_s)
a5 = shapiro(np.log(N_s))
a6 = shapiro(LinN_s)
a7 = shapiro(np.log(LinN_s))
#a4 = shapiro(np.log(NormData))

print(a4,a5,a6,a7)
#%%
#plt.plot(np.sort(NormData))
plt.figure()
#plt.hist(NormData,alpha=0.54)
plt.hist(LogData,alpha=0.6)
NormMu ,NormStd = norm.fit(NormData)#, kwds)

#%%
#LogLikely = np.prod(lognorm.pdf(LogData,LogSigma,scale=np.exp(LogMu)))
LogLikely = (lognorm.pdf(LogData,LogSigma,scale=np.exp(LogMu)))
NormLikely = np.prod(norm.pdf(NormData,NormMu,NormStd))

WrongLogLikely = np.prod(lognorm.pdf(NormData,LogSigma,scale=np.exp(LogMu)))
WrongNormLikely = np.prod(norm.pdf(LogData,NormMu,NormStd))

print(f"Log{LogLikely} other {WrongLogLikely}")
print(f"Norm{NormLikely} other {WrongNormLikely}")