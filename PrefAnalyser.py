# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:48:37 2024

@author: farka
"""

#import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


Data = pd.read_csv("100Langs.csv", decimal=',')
SpeakerNumbers = (Data["Total Speakers"]).to_numpy()


for i in range(1,2):
    plt.figure()#,figsize=(10,10))
    
    BluePebbleResult = ResultsFor("Pref", 0.3+0.1*i, 40)
    OtherArray =np.zeros(256)
    for j in BluePebbleResult.files:
    
        Result = LanguageVector(BluePebbleResult[j])
        OtherArray += Result
        plt.plot(Result,label=f"Attempt{j}")
        
    plt.xlabel("n languages")
    plt.ylabel("n speakers")
    plt.title(f"Language distribution for T={0.3+0.1*i}")
    FreqName = f"PrefResultForT{0.3+0.1*i}.jpeg"
    HistName = f"PrefHistForT{0.3+0.1*i}.jpeg"
    plt.savefig(FreqName,bbox_inches='tight', dpi=150,)
    #plt.figure(dpi=100, figsize=(10,10))
    plt.figure()
    plt.hist(OtherArray, bins=20)
    #plt.title("Language distribution for")
    plt.title(f"Language distribution for T={0.3+0.1*i}")
    plt.xlabel("N speaker")
    plt.ylabel("$n_s$ number of languages")
    plt.savefig(HistName,bbox_inches='tight', dpi=150,)


#print(BluePebbleResult.files)
#BitLangDist = LanguageDist(SimulatedArray)
#BitSpeakerNumbers = np.array(list(BitLangDist.values()))
#plt.figure()
#plt.loglog(SpeakerNumbers)
#plt.loglog(BitSpeakerNumbers)
#plt.figure()
#plt.plot(BitSpeakerNumbers)

#print("Current %d, Peak %d" %tracemalloc.get_traced_memory())