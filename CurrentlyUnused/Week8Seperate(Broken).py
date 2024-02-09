# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:23:06 2023

@author: farka
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random


global Temp
global GridSize 
global NLangFeatures
global GeneralThreshold
global NCounter

##Simulation Paramaters
GridSize = 40
NLangFeatures = 4
Temp = 0.1
NTimeSteps = 4000000
NFrames = 10
GeneralThreshold = 0.5*NLangFeatures
StepsPerFrame = math.floor(NTimeSteps/NFrames)
Ones=np.ones(NLangFeatures)
NCounter = 0
CounterConstant = math.floor(NTimeSteps/(NFrames*(NFrames+1)))

subtitle_string = f"With Grid Length { GridSize }, Language Vector Length {NLangFeatures},\n and Temperature {Temp}"

from Week8Init import *
from Week8Metro import *


Population = LatticeGenerate(NLangFeatures)

DistanceFreqInitial = EdgeDistanceDist()
LanguageDist()

Time,NRG = Metropolis(NTimeSteps) 

DistanceFreq = EdgeDistanceDist()
plt.figure()
plt.plot(Time,NRG)
plt.title("Normalised Energy time relationship")
plt.figtext(0.5, -0.3, subtitle_string , wrap=True, horizontalalignment='center', fontsize=8)
plt.figure()
plt.bar([i for i in range(0,NLangFeatures+1)],DistanceFreq)
plt.title("Distribution of Distance Between Neighbours")
plt.figtext(0.5, -0.3, subtitle_string , wrap=True, horizontalalignment='center', fontsize=8)
LanguageDist()