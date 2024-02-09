# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:48:37 2024

@author: farka
"""

#import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BluePebResult = np.load("BluePebbleOut.npz")
#prin
SimulatedArray = BluePebResult['arr_0']
TestingPassBack = BluePebResult['arr_1']
[GridSize,Temp,NTimeSteps,NLangFeatures] = [i for i in TestingPassBack]



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


Data = pd.read_csv("100Langs.csv", decimal=',')
SpeakerNumbers = (Data["Total Speakers"]).to_numpy()

BitLangDist = LanguageDist(SimulatedArray)
BitSpeakerNumbers = np.array(list(BitLangDist.values()))

plt.loglog(SpeakerNumbers)
plt.loglog(BitSpeakerNumbers)
plt.figure()
plt.plot(BitSpeakerNumbers)