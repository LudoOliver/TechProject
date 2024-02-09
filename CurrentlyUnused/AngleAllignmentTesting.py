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
GridSize = 20
NLangFeatures = 2
Temp = 0.05
NTimeSteps = 500000
NFrames = 30
StepsPerFrame = math.floor(NTimeSteps/NFrames)
Ones=np.ones(NLangFeatures)
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
    
def Energy():
    SumSum = 0
    for i in range(0,GridSize**2):
        AgentX = 0
        XNeighbours = SpeakerNeighbour(i)
        for j in XNeighbours:
            AgentX += np.dot(Population[i].Spin,Population[j].Spin)
        SumSum += AgentX
    return SumSum#*(-1)/(5*(GridSize**2)*NLangFeatures)


def SimpleDeltaE(x):

    XNeighbours = SpeakerNeighbour(x)
    YSum = np.zeros(NLangFeatures)
    for i in XNeighbours:
        YSum = np.add(Population[i].Spin,YSum)
    DE= 2*np.dot(Population[x].Spin,YSum)/len(XNeighbours)
    if DE <= 0:
        Population[x].Spin = Population[x].Spin*-1
    elif random.random() < math.exp((-DE)/Temp):
        Population[x].Spin = Population[x].Spin*-1
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
            
    LanguageNames = list(SpinDict.keys())
    LanguageFrequency = list(SpinDict.values())
    plt.figure()
    plt.bar(LanguageNames,LanguageFrequency)
    return SpinDict

def Metropolis(TimeSteps):
    
    SpinVisualiser()
    y=[]
    for i in range(1,TimeSteps):
        SimpleDeltaE(random.randint(0, GridSize**2-1))
        #if i%10==0:
            #y.append(Energy())
        if i%(StepsPerFrame)==0:
            SpinVisualiser()
    x = [x for x in range(len(y))]
    return(x,y)

def SpinVisualiser():
    

    """ Currently only works for 1d spin"""
    zs=np.zeros((GridSize,GridSize))
    plt.style.use('_mpl-gallery-nogrid')
    for i in range(0,len(Population)):
        a,b = Indice2Pos(i)
        zs[a-1,b-1]=(np.sum(Population[i].Spin-Ones))
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(zs)
    plt.show()
    

Population = LatticeGenerate(NLangFeatures)

Metropolis(NTimeSteps) 
#Without evluating energy up to about 3e5 is relatively quick




