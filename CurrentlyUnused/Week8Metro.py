# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:05:21 2023

@author: farka
"""

#import torch as t
import math
import numpy as np
import matplotlib.pyplot as plt
import random

def SimpleDeltaE(x): ##Made to only change one spin

    XNeighbours = SpeakerNeighbour(x)
    YSum = 0
    SpinToChange = random.randint(0, NLangFeatures-1)
    for i in XNeighbours:
        YSum += (Population[i].Spin)[SpinToChange]
    DE= 2*((Population[x].Spin)[SpinToChange])*YSum/len(XNeighbours)
    if DE <= 0:
        Population[x].Spin[SpinToChange] *= -1
    elif random.random() < math.exp((-DE)/Temp):
        Population[x].Spin[SpinToChange] *= -1
    return

def ExclusionDeltaE(x): ##Made to only change one spin

    XNeighbours = [NotLeastAlligned(x)]
    YSum = 0
    SpinToChange = random.randint(0, NLangFeatures-1)
    for i in XNeighbours:
        YSum += (Population[i].Spin)[SpinToChange]
    DE= 2*((Population[x].Spin)[SpinToChange])*YSum/len(XNeighbours)
    if DE <= 0:
        Population[x].Spin[SpinToChange] *= -1
    elif random.random() < math.exp((-DE)/Temp):
        Population[x].Spin[SpinToChange] *= -1
    return
def ThresholdDeltaE(x): ##Made to only change one spin

    XNeighbours = ThresholdNeighbours(x)
    if not XNeighbours:
        XNeighbours = SpeakerNeighbour(x)
    YSum = 0
    SpinToChange = random.randint(0, NLangFeatures-1)
    for i in XNeighbours:
        YSum += (Population[i].Spin)[SpinToChange]
    DE= 2*((Population[x].Spin)[SpinToChange])*YSum/len(XNeighbours)
    if DE <= 0:
        Population[x].Spin[SpinToChange] *= -1
    elif random.random() < math.exp((-DE)/Temp):
        Population[x].Spin[SpinToChange] *= -1
        
    return

def PreffenceDeltaE(x): ##Made to only change one spin

    Closest = MostAlligned(x)
    SpinToChange = random.randint(0, NLangFeatures-1)
    DE= 2*((Population[x].Spin)[SpinToChange])*Population[Closest].Spin[SpinToChange]
    if DE <= 0:
        Population[x].Spin[SpinToChange] *= -1
    elif random.random() < math.exp((-DE)/Temp):
        Population[x].Spin[SpinToChange] *= -1
    return

def MostAlligned(x):
    XNeighbours = (SpeakerNeighbour(x))
    np.random.shuffle(XNeighbours)
    Fitness = []
    for i in XNeighbours:
        Fitness.append(np.count_nonzero(Population[i].Spin!=Population[x].Spin))
    BestNeighbours = XNeighbours[(Fitness.index(min(Fitness)))]
    return BestNeighbours

def NotLeastAlligned(x):
    XNeighbours = (SpeakerNeighbour(x))
    np.random.shuffle(XNeighbours)
    Fitness = []
    for i in XNeighbours:
        Fitness.append(np.count_nonzero(Population[i].Spin!=Population[x].Spin))
    BestNeighbours = XNeighbours.pop(Fitness.index(min(Fitness)))
    return BestNeighbours

def ThresholdNeighbours(x):
    XNeighbours = (SpeakerNeighbour(x))
    return [i for i in XNeighbours 
            if np.count_nonzero(Population[i].Spin!=Population[x].Spin)<GeneralThreshold]
    
# def SimpleDeltaE(x): this spins the whole thing

def LocalisedDeltaE(x): #no current consideration of temperature in this model
    XNeighbours = SpeakerNeighbour(x)
    YSum = np.zeros(NLangFeatures)
    for i in XNeighbours:
        YSum = np.add(Population[i].Spin,YSum)
    DiffVec = 1-np.abs(Population[x].Spin-np.sign(YSum-0.1*Population[x].Spin))
    ## Current 0.1 needs changing to avoid bias to + spin
    Population[x].Spin= np.multiply(Population[x].Spin, DiffVec)
    #if DE <= 0:
        #Population[x].Spin = Population[x].Spin*-1
    #elif random.random() < math.exp((-DE)/Temp):
        #Population[x].Spin = Population[x].Spin*-1
    return




def Metropolis(TimeSteps):
    StepCounter = 1
    StepValue = CounterConstant
    SpinVisualiser()
    y=[Energy()]
    for i in range(1,TimeSteps):
        ThresholdDeltaE(random.randint(0, GridSize**2-1))
        #if i%10==0:
            #y.append(Energy())
        # if i%(StepsPerFrame)==0:
        #     SpinVisualiser()
        #     y.append(Energy())
        if i == StepValue:
            SpinVisualiser()
            y.append(Energy())
            StepValue += (2*StepCounter+3)*CounterConstant
            StepCounter+=1
    x = [x for x in range(len(y))]
    return(x,y)

