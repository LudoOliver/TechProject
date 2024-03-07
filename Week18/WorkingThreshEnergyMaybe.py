# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:42:09 2024

@author: xd21736
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
#import time
import cProfile
from scipy.stats import lognorm , norm, shapiro
from scipy import stats

import statsmodels.api as sm 
import pylab #as py 

global Temp
global GridSize 
global NLangFeatures
global GeneralThreshold
global NCounter

GridSize = 300
NLangFeatures = 8
Temp = 0 
Ones=np.ones(NLangFeatures)


N=40
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0.1,0.8,N)))
##Simulation Paramaters
GridSize = 300
NLangFeatures = 8
Temp = 0.1
NTimeSteps = int(1e6) #was 1e7
NFrames = 30
GeneralThreshold = 0.5*NLangFeatures
StepsPerFrame = math.floor(NTimeSteps/NFrames)
Ones=np.ones(NLangFeatures)


ConvergenceThreshold = 0.001 #was 0.01 for large scale

##Counter variables
NCounter = 0
CounterConstant = math.floor(NTimeSteps/(NFrames*(NFrames+1)))
#MultiTemp variables
TimesEvaluated = 15
LowerBound = 0.001 #nornally 0.0001
UpperBound = 2#normally 1
TempValues = np.linspace(LowerBound, UpperBound,TimesEvaluated)
#EnableGraphs
GraphFlag = 1
subtitle_string = f"Evaulated for L ={ GridSize }, {NLangFeatures} Spins ,over {NTimeSteps} Steps"

def Indice2Pos(x):
    j = x%GridSize+1
    i = math.floor(x/GridSize)+1
    return i,j

def Pos2Indice(pos):
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
    def __init__(self,Index,SpinArray):
        #self.
        self.Indice = Index
        self.Spin = SpinArray
        self.Neighbours = SpeakerNeighbour(Index)
        
    def MostAlligned(self):
        #np.random.shuffle(XNeighbours)
        Fitness = []
        for i in self.Neighbours:
            Fitness.append(np.count_nonzero(Population[i].Spin!=self.Spin))
        BestNeighbours = self.Neighbours[(Fitness.index(min(Fitness)))]
        return BestNeighbours
    
    def PreffenceDeltaE(self): 
        Closest = self.MostAlligned()
        SpinToChange = random.randint(0, NLangFeatures-1)
        #DE= 2*((self.Spin)[SpinToChange])*Population[Closest].Spin[SpinToChange]
        if self.Spin[SpinToChange] !=  Population[Closest].Spin[SpinToChange]:
            self.Spin[SpinToChange] *= -1
        elif random.random() < math.exp((-2)/Temp):
            self.Spin[SpinToChange] *= -1
        return
    
    def SimpleDeltaE(self): 
        YSum = 0
        SpinToChange = random.randint(0, NLangFeatures-1)
        for i in self.Neighbours :
            YSum += (Population[i].Spin)[SpinToChange]
        DE= 2*((self.Spin)[SpinToChange])*YSum/len(self.Neighbours)
        if DE <= 0:
            self.Spin[SpinToChange] *= -1
        elif random.random() < math.exp((-DE)/Temp):
            self.Spin[SpinToChange] *= -1
        return
    
    def ThresholdNeighbours(self):
        #XNeighbours = (SpeakerNeighbour(x))
        return [i for i in self.Neighbours
                if np.count_nonzero(Population[i].Spin!=self.Spin)<GeneralThreshold]
    
    def ThresholdDeltaE(self): ##Made to only change one spin

        XNeighbours = self.ThresholdNeighbours()
        if not XNeighbours:
            XNeighbours = self.Neighbours
        YSum = 0
        SpinToChange = random.randint(0, NLangFeatures-1)
        for i in XNeighbours:
            YSum += (Population[i].Spin)[SpinToChange]
        DE= 2*((self.Spin)[SpinToChange])*YSum/len(XNeighbours)
        if DE <= 0:
            self.Spin[SpinToChange] *= -1
        elif random.random() < math.exp((-DE)/Temp):
            self.Spin[SpinToChange] *= -1
        
        return
    

    
        
def LatticeGenerate(SinglePopMatrix):
    
    Lattice = []
    for i in range(0,GridSize**2):
        Lattice.append(Speaker(i, SinglePopMatrix[i,:]))
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


def FastEnergy():
    SumSum = 0
    for i in range(0,GridSize**2):
        #AgentX = 0
        #XNeighbours = SpeakerNeighbour(i)
        for j in Population[i].Neighbours:
            SumSum += np.dot(Population[i].Spin,Population[j].Spin)
    return SumSum*(-1)/(4*(GridSize**2)*NLangFeatures)




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
    




def PreffMetro(TimeSteps):
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
    print(f"Prefference failed to converge in {NTimeSteps} time steps, with L= {GridSize}")
    return FastEnergy()

def ThreshMetro(TimeSteps):
    #StepCounter = 1
    #StepValue = CounterConstant
    #OldEnergy= 10
    EnergyArray = [10,10,10]
    for i in range(1,TimeSteps):
        Population[random.randint(0, GridSize**2-1)].ThresholdDeltaE()
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
    print(f"Threshold failed to converge in {NTimeSteps}steps, with L= {GridSize}")
    return FastEnergy()

def SimpleMetro(TimeSteps):
    #StepCounter = 1
    #StepValue = CounterConstant
    #OldEnergy= 10
    EnergyArray = [10,10,10]
    for i in range(1,TimeSteps):
        Population[random.randint(0, GridSize**2-1)].SimpleDeltaE()
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
    print(f"Simple failed to converge in {NTimeSteps}steps, with L= {GridSize}")
    return FastEnergy()


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
    plt.xlabel("Language")
    plt.ylabel("Frequency")
    plt.figtext(0.5, -0.3, subtitle_string , wrap=True, horizontalalignment='center', fontsize=8)
    return SpinDict

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
    plt.figure()
    plt.bar([i for i in range(0,NLangFeatures+1)],DistanceFreq)
    plt.title("Distance Between Neighbouring Nodes")
    plt.xticks(ticks=[i for i in range(0,NLangFeatures+1)])
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.figtext(0.5, -0.3, subtitle_string , wrap=True, horizontalalignment='center', fontsize=8)
    return DistanceFreq

def MakeArray():
    return np.array([i.Spin for i in Population])
    


def MakeNormal(InArray):
    Top = max(InArray)
    Bottom = min(InArray)
    Output = [(i-Bottom)/(Top-Bottom)-1 for i in InArray]
    return Output

def ResultsFor(mode,temperature,length):
    FileName = f"{mode}L{length}T{temperature:.2f}.npz"
    print(FileName)
    DataInFile = np.load(FileName)
    return DataInFile

def LatticeGenerate(SinglePopMatrix): 
    Lattice = []
    for i in range(0,GridSize**2):
        Lattice.append(Speaker(i, SinglePopMatrix[i,:]))
    return Lattice
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
#%%
#TempRange = [0.015*i for i in range(20,40)]+[0.1*i for i in range(3,8)]
#TempRange.sort()
ThreshTempValues = [0.015*i for i in range(1,55)]
ThreshTempLabels = [f"{i:.2f}" for i in ThreshTempValues]

TempRange = ThreshTempLabels
# def LatticeGenerate(SinglePopMatrix,GridSize=300):
    
#     Lattice = []
#     for i in range(0,GridSize**2):
#         Lattice.append(Speaker(i, SinglePopMatrix[i,:]))
#     return Lattice
ThreshBigArray = np.zeros([len(TempRange),256])
plt.figure()   

EnergyMatrix = np.zeros([len(TempRange),15])+np.nan
for i in range(0,len(TempRange)):
    Data = ResultsFor('Thresh', float(TempRange[i]), 300)
    for j in range(0,len(Data.files)):
        
        ThreshBigArray[i,:] += LanguageVector(Data[str(j)])
        #OtherArray += Result
    #     plt.plot(Result,label=f"Attempt{j}")
     #= OtherArray
        Population = 0
        Population = LatticeGenerate(Data[str(j)])
        #SpinVisualiser()
        E = FastEnergy()
        print(E)
        EnergyMatrix[i,j] = E
        #print(f"{i*10+j}% done")
plt.cla()
EnergyVector= np.nanmean(EnergyMatrix,axis=1) 
for i in range(15):
    #plt.plot(TempRange, EnergyMatrix[:,i],alpha=0.4)
    stats.probplot(ThreshBigArray[i,:], dist="norm", plot=pylab)
plt.plot(ThreshTempValues ,EnergyVector,color='b')   
plt.xlabel("Temperature")
plt.ylabel("Energy")
plt.yticks([])
#plt.xlim((0.4,0.7))#, kwargs)
#plt.ylim((-0.55,-0.2))
plt.title("Preference Phase Behaviour")

stats.probplot(ThreshBigArray, dist="norm", plot=pylab)

#%%
plt.figure()  
for j in range(0,len(TempRange)):
    OtherArray = ThreshBigArray[j,:]     
    n,x = np.histogram(OtherArray, bins=20, density=True)
        #plt.title("Language distribution for")
    density = stats.gaussian_kde(OtherArray)
    plt.plot(x,density(x),label=f"T={TempRange[i]}",alpha=0.1)
    #plt.fill_between(x,density(x),alpha=0.4)
    #plt.loglog(x,density(x),label=f"T={0.3+0.1*j:.2f}",alpha=1-j*0.1)
   
DistName = "ThreshDistForLogLog.jpeg"
#plt.legend()
plt.xticks([],[])
plt.yticks([],[])
plt.title("Language Distribution of the Thresh Model")
plt.xlabel("N speakers")
plt.ylabel("$n_s$ number of languages")
#plt.savefig(DistName,bbox_inches='tight', dpi=300)

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
#x = stats.loggamma.rvs(c=2.5, size=500, random_state=rng)
#res = stats.probplot(x, dist=stats.loggamma, sparams=(2.5,), plot=ax)
for i in range(15):
    #plt.plot(TempRange, EnergyMatrix[:,i],alpha=0.4)
    stats.probplot(PrefBigArray[i,:], dist="norm", plot=ax)
ax.set_title("QQ plot of Prefence model across varying Temp")



