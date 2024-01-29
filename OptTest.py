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
import time
import cProfile
global Temp
global GridSize 
global NLangFeatures
global GeneralThreshold
global NCounter
##Simulation Paramaters
GridSize = 400
NLangFeatures = 8
Temp = 0.7
NTimeSteps = int(1e6) #was 1e7
NFrames = 30
GeneralThreshold = 0.5*NLangFeatures
StepsPerFrame = math.floor(NTimeSteps/NFrames)
Ones=np.ones(NLangFeatures)
"""Notes
Conor goes to 2 as limit for temperature
Investigate wtf is going on with simple delta e
"""
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

class Speaker():
    def __init__(self,Index,NLangFeatures):
        self.Indice = Index
        self.Spin = np.random.choice(a=[-1, 1], size=(NLangFeatures))
        self.Neighbours = SpeakerNeighbour(Index)
    def MostAlligned(self):
        #np.random.shuffle(XNeighbours)
        Fitness = []
        for i in self.Neighbours:
            Fitness.append(np.count_nonzero(Population[i].Spin!=self.Spin))
        BestNeighbours = self.Neighbours[(Fitness.index(min(Fitness)))]
        return BestNeighbours
    def PreffenceDeltaE(self): ##Made to only change one spin
        Closest = self.MostAlligned()
        SpinToChange = random.randint(0, NLangFeatures-1)
        #DE= 2*((self.Spin)[SpinToChange])*Population[Closest].Spin[SpinToChange]
        if self.Spin[SpinToChange] !=  Population[Closest].Spin[SpinToChange]:
            self.Spin[SpinToChange] *= -1
        elif random.random() < math.exp((-2)/Temp):
            self.Spin[SpinToChange] *= -1
        return

        #, p=[0.99,0.01]) tests that energy does reflect dgeree of randomisation
        
def LatticeGenerate(NLangFeatures):
    
    Lattice = []
    for i in range(0,GridSize**2):
        Lattice.append(Speaker(i,NLangFeatures))
    return Lattice



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
def FastEnergy():
    SumSum = 0
    for i in range(0,GridSize**2):
        #AgentX = 0
        #XNeighbours = SpeakerNeighbour(i)
        for j in Population[i].Neighbours:
            SumSum += np.dot(Population[i].Spin,Population[j].Spin)
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
# def PreffenceDeltaE(x): ##Made to only change one spin

#     Closest = MostAlligned(x)
#     SpinToChange = random.randint(0, NLangFeatures-1)
#     DE= 2*((Population[x].Spin)[SpinToChange])*Population[Closest].Spin[SpinToChange]
#     if DE <= 0:
#         Population[x].Spin[SpinToChange] *= -1
#     elif random.random() < math.exp((-DE)/Temp):
#         Population[x].Spin[SpinToChange] *= -1
#     return


# def MostAlligned(x):
#     XNeighbours = (SpeakerNeighbour(x))
#     np.random.shuffle(XNeighbours)
#     Fitness = []
#     for i in XNeighbours:
#         Fitness.append(np.count_nonzero(Population[i].Spin!=Population[x].Spin))
#     BestNeighbours = XNeighbours[(Fitness.index(min(Fitness)))]
#     return BestNeighbours

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
#@profile
def Metropolis(TimeSteps):
    #StepCounter = 1
    #StepValue = CounterConstant
    #OldEnergy= 10
    EnergyArray = [10,10,10]
    for i in range(1,TimeSteps):
        PreffenceDeltaE(random.randint(0, GridSize**2-1))
        if i%StepsPerFrame==0:
            #SpinVisualiser()
            NewEnergy=Energy()
            MeanEnergy = sum(EnergyArray[-3:])/3
            if abs(MeanEnergy-NewEnergy) <0.01:
                print(f"Settled after {i} steps")
                return NewEnergy
            EnergyArray.append(NewEnergy)
            #StepValue += (2*StepCounter+3)*CounterConstant
            #StepCounter+=1
    print(f"Failed to converge after {NTimeSteps} time steps, with grid size {GridSize}")
    return Energy()

def ClassPrefMet(TimeSteps):
    #StepCounter = 1
    #StepValue = CounterConstant
    #OldEnergy= 10
    EnergyArray = [10,10,10]
    for i in range(1,TimeSteps):
        Population[random.randint(0, GridSize**2-1)].PreffenceDeltaE()
        if i%StepsPerFrame==0:
            #SpinVisualiser()
            NewEnergy= FastEnergy()
            MeanEnergy = sum(EnergyArray[-3:])/3
            if abs(MeanEnergy-NewEnergy) <0.01:
                print(f"Settled after {i} steps")
                return NewEnergy
            EnergyArray.append(NewEnergy)
            #StepValue += (2*StepCounter+3)*CounterConstant
            #StepCounter+=1
    print(f"Failed to converge after {NTimeSteps} time steps, with grid size {GridSize}")
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
UpperBound = 2#normally 1
TempValues = np.linspace(LowerBound, UpperBound,TimesEvaluated)

def MakeNormal(InArray):
    Top = max(InArray)
    Bottom = min(InArray)
    Output = [(i-Bottom)/(Top-Bottom)-1 for i in InArray]
    return Output
GridSize = 30
tic = time.time()

pr = cProfile.Profile()
pr.enable()

Population = LatticeGenerate(NLangFeatures)
ClassPrefMet(NTimeSteps)

pr.disable()
pr.print_stats(sort = "cumtime")

toc = time.time()

print(toc-tic)
print(Energy())
print(FastEnergy())
# for k in range(1,2): #should be 6, alpha as .2*k
#     MeanEnergy = np.zeros(TimesEvaluated)
#     GridSize= 240 #should be k*60
#     for i in range(0,TimesEvaluated):
#         print(math.floor(100*(TimesEvaluated*(k-1)+i)/(5*TimesEvaluated)),'% Complete')
#         Temp = TempValues[i]
#         NRG =0
#         for j in range(1,2):
#             Population = LatticeGenerate(NLangFeatures)
#             NRG += 0.5*Metropolis(NTimeSteps)
#         MeanEnergy[i] = NRG
#     plt.plot(TempValues,MakeNormal(MeanEnergy), alpha=1, label=f"Gridsize ={GridSize}")
# plt.legend()
# plt.xlabel("Temperature")
# plt.ylabel("System Energy")
# plt.title("Prefference Phase Behaviour")
# #plt.figtext(0.5, -0.3, subtitle_string , wrap=True, horizontalalignment='center', fontsize=8)
# #Without evluating energy up to about 3e5 is relatively quick

""" Energy Temperature relationship"""


