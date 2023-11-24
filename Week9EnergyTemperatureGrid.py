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
global Temp
global GridSize 
global NLangFeatures
global GeneralThreshold
global NCounter
##Simulation Paramaters
GridSize = 200
NLangFeatures = 6
Temp = 0.3
NTimeSteps = 1000000
NFrames = 10
GeneralThreshold = 0.5*NLangFeatures
StepsPerFrame = math.floor(NTimeSteps/NFrames)
Ones=np.ones(NLangFeatures)

##Counter variables
NCounter = 0
CounterConstant = math.floor(NTimeSteps/(NFrames*(NFrames+1)))

subtitle_string = f"Threshold Model With \n Grid Length { GridSize },Language Vector Length {NLangFeatures},\n over {NTimeSteps} Timesteps"

def Indice2Pos(x):
    j = x%GridSize+1
    i = math.floor(x/GridSize)+1
    return i,j

def Pos2Indice(pos):
    #i,j = pos
    #x= i*GridSize+j-1
    x= (pos[0]-1)*GridSize+pos[1]-1
    return x

class Speaker():
    def __init__(self,Index,NLangFeatures):
        self.Indice = Index
        self.Spin = np.random.choice(a=[-1, 1], size=(NLangFeatures))
        #, p=[0.99,0.01]) tests that energy does reflect dgeree of randomisation
        
def LatticeGenerate(NLangFeatures):
    
    Lattice = []
    for i in range(0,GridSize**2):
        Lattice.append(Speaker(i,NLangFeatures))
    return Lattice

def SpeakerNeighbour(indice):
    NeighbourIndices= ([indice-GridSize-1,indice-GridSize,indice-GridSize+1,
                       indice-1,indice+1,indice+GridSize-1,indice+GridSize,
                       indice+GridSize+1])
    i,j = Indice2Pos(indice)
    if i not in [1,GridSize] and j not in [1,GridSize]:
        return [NeighbourIndices[i] for i in [1,3,4,6]]
    if i ==1 and j not in [1,GridSize]:
        return [NeighbourIndices[i] for i in [3,4,6]]
    if i ==GridSize and j not in [1,GridSize]:
        return [NeighbourIndices[i] for i in [1,3,4]]
    if j ==1 and i not in [1,GridSize]:
        return [NeighbourIndices[i] for i in
                    [1,4,6]]
    if j ==GridSize and i not in [1,GridSize]:
        return [NeighbourIndices[i] for i in 
                [1,3,6]]
    if j ==GridSize and i == GridSize:
        return [NeighbourIndices[i] for i in
                [1,3]]
    if j ==GridSize and i== 1:
        return [NeighbourIndices[i] for i in [3,6]]
    if j ==1 and i ==1:
        return [NeighbourIndices[i] for i in
                [4,6]]
    if j==1 and i ==GridSize:
        return [NeighbourIndices[i] for i in
                [1,4]]

def SideIndices(indice):
    i,j = Indice2Pos(indice)
    PossibleNeighbours = ([indice+GridSize,indice+1])
    if i==GridSize and j==GridSize:
        return 0
    elif i ==GridSize:
        return [indice+1]
    elif j ==GridSize:
        return [indice+GridSize]
    else:
        return PossibleNeighbours
def Energy():
    SumSum = 0
    for i in range(0,GridSize**2):
        AgentX = 0
        XNeighbours = SpeakerNeighbour(i)
        for j in XNeighbours:
            AgentX += np.dot(Population[i].Spin,Population[j].Spin)
        SumSum += AgentX
    return SumSum*(-1)/(4*(GridSize**2)*NLangFeatures)


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


def LanguageDist():
    BigList=[]
    for i in Population:
        BigList.append(np.array2string(i.Spin))
    SpinDict = {}
    for j in BigList:
        if j in SpinDict:
            SpinDict[j] += 1
        else:
            SpinDict.update({j: 1})
    SpinDict = dict(sorted(SpinDict.items(),key=lambda item: item[1],reverse=True))
    
    #LanguageNames = list(SpinDict.keys())
    LanguageFrequency = list(SpinDict.values())
    plt.figure()
    plt.plot(LanguageFrequency)
    plt.title("Language Frequency Distribution")
    plt.figtext(0.5, -0.3, subtitle_string , wrap=True, horizontalalignment='center', fontsize=8)
    return SpinDict

def Metropolis(TimeSteps):
    StepCounter = 1
    StepValue = CounterConstant
    for i in range(1,TimeSteps):
        ThresholdDeltaE(random.randint(0, GridSize**2-1))
    return Energy()

def SpinVisualiser():
    
    #will have to update not to switch colors

    """ Currently only works for 1d spin"""
    zs=np.zeros((GridSize,GridSize))
    plt.style.use('_mpl-gallery-nogrid')
    #plt.style.use('classic')
    for i in range(0,len(Population)):
        a,b = Indice2Pos(i)
        zs[a-1,b-1]=(np.sum(Population[i].Spin-Ones))
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(zs,cmap='viridis')
    plt.show()
    
def TestSomeSpins():
    for i in [7,13,15,19,78]:
        print(Population[i].Spin)

       
def EdgeDistanceDist(): ##Working In current form
    DistanceFreq = np.zeros(NLangFeatures+1)
    for i in range(0,len(Population)):
        Neighbours = SideIndices(i)
        if Neighbours ==0:
            break
        else:
            for j in Neighbours:
                NodeDistance = np.count_nonzero(Population[i].Spin!=Population[j].Spin)
                DistanceFreq[NodeDistance] += 1
    return DistanceFreq

# Ungrouped = [i for i in range(0,len(Population))]
# def ClusterDist(indice,Ungrouped):
#     PossibleNeighbours = NeighbourIndices(indice)
#     for i in 
    
TimesEvaluated = 15
LowerBound = 0.001 #nornally 0.0001
UpperBound = 1#normally 1
TempValues = np.linspace(LowerBound, UpperBound,TimesEvaluated)

GridSize =0
for k in range(1,6):
    MeanEnergy = np.zeros(TimesEvaluated)
    GridSize=k*60
    for i in range(0,TimesEvaluated):
        print(math.floor(100*(TimesEvaluated*(k-1)+i)/(5*TimesEvaluated)),'% Complete')
        Temp = TempValues[i]
        NRG =0
        for j in range(1,3):
            Population = LatticeGenerate(NLangFeatures)
            NRG += 0.5*Metropolis(NTimeSteps)
        MeanEnergy[i] = NRG
    plt.plot(TempValues,MeanEnergy, alpha=min(1,0.2*k))

plt.xlabel("Temperature")
plt.ylabel("System Energy")
#plt.figtext(0.5, -0.3, subtitle_string , wrap=True, horizontalalignment='center', fontsize=8)
#Without evluating energy up to about 3e5 is relatively quick

""" Energy Temperature relationship"""


