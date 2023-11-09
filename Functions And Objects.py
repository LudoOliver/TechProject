# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:05:21 2023

@author: farka
"""

import torch as t
import math
import numpy as np
import matplotlib.pyplot as plt

global GridSize 
global NLangFeatures
GridSize = 10
NLangFeatures = 6

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
        return NeighbourIndices
    if i ==1 and j not in [1,GridSize]:
        return [NeighbourIndices[i] for i in range(3,7)]
    if i ==GridSize and j not in [1,GridSize]:
        return [NeighbourIndices[i] for i in range(0,4)]
    if j ==1 and i not in [1,GridSize]:
        return [NeighbourIndices[i] for i in
                    [1,2,4,6,7]]
    if j ==GridSize and i not in [1,GridSize]:
        return [NeighbourIndices[i] for i in 
                [0,1,3,5,6]]
    if j ==GridSize and i == GridSize:
        return [NeighbourIndices[i] for i in
                [0,1,3]]
    if j ==GridSize and i== 1:
        return [NeighbourIndices[i] for i in [3,5,6]]
    if j ==1 and i ==1:
        return [NeighbourIndices[i] for i in
                [4,6,7]]
    if j==1 and i ==GridSize:
        return [NeighbourIndices[i] for i in
                [1,2,4]]
    else:
        return "Retard"
    
def Energy():
    SumSum = 0
    for i in range(0,GridSize**2):
        AgentX = 0
        XNeighbours = SpeakerNeighbour(i)
        for j in XNeighbours:
            AgentX += np.dot(Population[i].Spin,Population[j].Spin)
        SumSum += AgentX
    return SumSum#*(-1)/(5*(GridSize**2)*NLangFeatures)

def DeltaE(x):
    ENought = Energy()
    DifferenceVector = np.random.choice(a=[-1, 1], size=(NLangFeatures), p=[0.1,0.9])
    XNeighbours = SpeakerNeighbour(x)
    ESum =0
    for i in XNeighbours:
        ESum += np.dot(DifferenceVector,Population[i].Spin)
    if ESum>0:
        return
    else:
        Population[x].Spin = np.multiply(Population[x].Spin,DifferenceVector)
        ENew = Energy()
        print(ENew-ENought)
        return
    
def NaiveDeltaE(x):
    ENought = Energy()
    DifferenceVector = np.random.choice(a=[-1, 1], size=(NLangFeatures),
                                        p=[0.1,0.9])
    NewSpin = np.multiply(Population[x].Spin,DifferenceVector)
    XNeighbours = SpeakerNeighbour(x)
    E1Sum =0
    E2Sum =0
    for i in XNeighbours:
        E1Sum += np.dot(Population[x].Spin,Population[i].Spin)
        E2Sum += np.dot(NewSpin,Population[i].Spin)

    Population[x].Spin = np.multiply(Population[x].Spin,DifferenceVector)
    ENew =Energy()

    return (E2Sum-E1Sum),(ENew-ENought)
    # if ESum>0:
    #     return
    # else:
    #     Population[x].Spin = np.multiply(Population[x].Spin,DifferenceVector)
    #     ENew = Energy()
    #     print(ENew-ENought)
    #     return    
    
Population = LatticeGenerate(NLangFeatures)

#DeltaE(6)

"""
Test for proportionality between change in energy and the
differnce in the inner p of differnce vector, proved roughly proportional
with a clear +ve corelation
x=[]
y=[]
for i in range(0,30):
    print(i)
    a,b = NaiveDeltaE(i)
    x.append(a)
    y.append(b)
plt.scatter(x, y,)""" # 
    
#print(Population[1].Spin)
#Agent = Population[1].Spin
#print(SpeakerNeighbour(6))
#print(Energy())
#i,j = Indice2Pos(0)
#Six = Pos2Indice([i,j])