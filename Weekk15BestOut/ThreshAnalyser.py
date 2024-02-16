# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:48:37 2024

@author: farka
"""

#import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import math



from matplotlib.collections import PolyCollection
#mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["r", "k", "c"]) 
N=12
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0.1,0.8,N)))

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


#Data = pd.read_csv("100Langs.csv", decimal=',')
#SpeakerNumbers = (Data["Total Speakers"]).to_numpy()


# for i in range(1,2):
#     plt.figure()#,figsize=(10,10))
    
#     BluePebbleResult = ResultsFor("Pref", 0.3+0.1*i, 40)
#     OtherArray =np.zeros(256)
#     for j in BluePebbleResult.files:
    
#         Result = LanguageVector(BluePebbleResult[j])
#         OtherArray += Result
#         plt.plot(Result,label=f"Attempt{j}")
        
#     plt.xlabel("n languages")
#     plt.ylabel("n speakers")
#     plt.title(f"Language distribution for T={0.3+0.1*i}")
#     FreqName = f"PrefResultForT{0.3+0.1*i}.jpeg"
#     HistName = f"PrefHistForT{0.3+0.1*i}.jpeg"
#     plt.savefig(FreqName,bbox_inches='tight', dpi=150,)
#     #plt.figure(dpi=100, figsize=(10,10))
#     plt.figure()
#     plt.hist(OtherArray, bins=20)
#     #plt.title("Language distribution for")
#     plt.title(f"Language distribution for T={0.3+0.1*i}")
#     plt.xlabel("N speaker")
#     plt.ylabel("$n_s$ number of languages")
#     plt.savefig(HistName,bbox_inches='tight', dpi=150,)
BigArray = np.zeros([10,256])
plt.figure()   
for i in range(6,16):
    #plt.figure()#,figsize=(10,10))
    
    BluePebbleResult = ResultsFor("Thresh", 0.05*i, 300)
    OtherArray =np.zeros(256)
    for j in BluePebbleResult.files:
    
        Result = LanguageVector(BluePebbleResult[j])
        OtherArray += Result
    #     plt.plot(Result,label=f"Attempt{j}")
    BigArray[i-6,:] = OtherArray
    # plt.xlabel("n languages")
    # plt.ylabel("n speakers")
    # plt.title(f"Language distribution for T={0.3+0.1*i:.2f}")
    # FreqName = f"PrefResultForT{0.3+0.1*i:.2f}.jpeg"
    # HistName = f"PrefHistForT{0.3+0.1*i:.2f}.jpeg"
    # plt.savefig(FreqName,bbox_inches='tight', dpi=150,)
    #plt.figure(dpi=100, figsize=(10,10))
    #plt.figure()
    n,x = np.histogram(OtherArray, bins=20, density=True)
    #plt.title("Language distribution for")
    density = stats.gaussian_kde(OtherArray)
    plt.plot(x,density(x),label=f"T={0.05*i:.2f}",alpha=0.9)
    #plt.loglog(x,density(x),label=f"T={0.3+0.1*i:.2f}",alpha=1-i*0.1)
    
DistName = "ThreshDistForVaryingT.jpeg"
plt.legend()
plt.xticks([],[])
plt.yticks([],[])
plt.title("Language Distribution of the Threshold Model")
plt.xlabel("N speakers")
plt.ylabel("$n_s$ number of languages")
#plt.savefig(DistName,bbox_inches='tight', dpi=300)
  
#%%
plt.figure()  
for j in range(6,16):
    OtherArray = BigArray[j,:]     
    n,x = np.histogram(OtherArray, bins=20, density=True)
        #plt.title("Language distribution for")
    density = stats.gaussian_kde(OtherArray)
    plt.plot(x,density(x),label=f"T={0.05*j:.2f}",alpha=0.9)
    plt.fill_between(x,density(x),alpha=0.4)
    #plt.loglog(x,density(x),label=f"T={0.3+0.1*j:.2f}",alpha=1-j*0.1)
   
DistName = "ThreshDistForLogLog.jpeg"
plt.legend()
plt.xticks([],[])
plt.yticks([],[])
plt.title("Language Distribution of the Threshold Model")
plt.xlabel("N speakers")
plt.ylabel("$n_s$ number of languages")
#plt.savefig(DistName,bbox_inches='tight', dpi=300)
#%%
"""Trying 3dNumber1"""

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x = np.linspace(-50,50,100)
# y = np.arange(25)
# X,Y = np.meshgrid(x,y)
# Z = np.zeros((len(y),len(x)))

# for i in range(len(y)):
#     damp = (i/float(len(y)))**2
#     Z[i] = 5*damp*(1 - np.sqrt(np.abs(x/50)))
#     Z[i] += np.random.uniform(0,.1,len(Z[i]))
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1000, color='w', shade=False, lw=.5)

# ax.set_zlim(0, 5)
# ax.set_xlim(-51, 51)
# ax.set_zlabel("Intensity")
# ax.view_init(20,-120)
# plt.show()
    
#%%
"""Trying 3dNumber2"""
    
def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]
#plt.figure()

ax = plt.figure(figsize=(10,10),constrained_layout=True).add_subplot(projection='3d')

#x = np.linspace(0., 10., 31)
lambdas = [(i*0.05) for i in range(6, 16)]

# verts[i] is a list of (x, y) pairs defining polygon i.
gamma = np.vectorize(math.gamma)
#verts = [polygon_under_graph(x, density(x)
         #for l in lambdas]
verts = []  
mval=0                         
for j in range(0,len(lambdas)):
    OtherArray = BigArray[j,:]     
    n,x = np.histogram(OtherArray, bins=20, density=True)
        #plt.title("Language distribution for")
    density = stats.gaussian_kde(OtherArray)     
    #verts.append(polygon_under_graph(x, density(x)))
    ax.plot(x,density(x),lambdas[j],color='k',zdir='y')
    ax.add_collection3d((plt.fill_between(x,density(x),alpha=0.5)), zs=lambdas[j], zdir='y')
                       
facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))

#poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
#ax.add_collection3d(poly, zs=lambdas, zdir='y')
#lambdas.insert(0,0)
ax.set( xlabel='N',ylabel='T')
#ax.xaxis.set_ticks(values_list)
ax.set_yticks(lambdas)
ax.set_yticklabels([f"{i:.2f}"for i in lambdas],fontsize=9,verticalalignment='baseline',horizontalalignment='left')#, zlabel='$n_s$')
ax.zaxis.set_rotate_label(False)
#ax.set_zlabel("Depth ($\mu$$m$)") # Updated

 # (+) Added
#ax.zaxis.set_label_coords(-10, -10)
#ax.zaxis._axinfo['label']['juggled'] = (1,2,0)
ax.set_zlabel('$n_s$',rotation=0,horizontalalignment='right',verticalalignment='baseline',fontsize=25, labelpad=-10)
ax.tick_params(axis="z", pad=-3)
ax.set_xlabel('N',rotation=0,horizontalalignment='right',verticalalignment='baseline',fontsize=25, labelpad=-10)
ax.tick_params(axis="x", pad=-3)
ax.set_ylabel('T',rotation=0,horizontalalignment='right',verticalalignment='baseline',fontsize=25, labelpad=10)
ax.tick_params(axis="y", pad=1)
#ax.zaxis._axinfo['label']['space_factor'] = 0
ax.zaxis._axinfo['label']['juggled'] = (1,2,0)
# Hide grid lines
ax.grid(False)
#ax.invert_yaxis()
# Hide axes ticks
ax.set_xticks([])
ax.set_title("Language Distribution across $T \in [0.3,0.75]$",fontsize=30)
ax.view_init(elev=30)
#ax.set_yticks(lambdas,labels="T",fontsize=5)
#ax.set_yticks([])
#ax.view_init(azim=100)
ax.set_zticks([])
#ax.dist=11
plt.savefig("Thresh3DLang.png",bbox_inches='tight', dpi=300)
#ax.xticks([],[])
#ax.yticks([],[])
#plt.show()    
#     #plt.figure()
# plt.figure()
# for i in range(0,5):
#     plt.loglog(BigArray[i,:]/10,label=f"T={0.3+0.1*i:.2f}")
# plt.legend()
# plt.xticks([],[])
# plt.yticks([],[])
# plt.xlabel("Language")
# plt.ylabel("Number of speakers")
# plt.title("Language distribution across varying temperatures")


#print(BluePebbleResult.files)
#BitLangDist = LanguageDist(SimulatedArray)
#BitSpeakerNumbers = np.array(list(BitLangDist.values()))
#plt.figure()
#plt.loglog(SpeakerNumbers)
#plt.loglog(BitSpeakerNumbers)
#plt.figure()
#plt.plot(BitSpeakerNumbers)

#print("Current %d, Peak %d" %tracemalloc.get_traced_memory())