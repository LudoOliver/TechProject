# -*- coding: utf-8 -*-
"""
Showing the two dists at t=0.57 aka i=37 for thresh

and for pref this is i=18 (probably)
Created on Mon Mar 25 18:07:46 2024

@author: Admin
"""
import math
import numpy as np
import matplotlib.pyplot as plt
#First lets do layered energy plots
PrefBigArray = np.load("FPrefDistMatrix.npy")
PrefEnergyArray = np.load("FPrefEnergyMatrix.npy")
ThreshBigArray = np.load("ThreshBigArrayFinal.npy")
ThreshEnergyArray = np.load("ThreshEnergyFinal.npy")

TempValues = [0.015*i for i in range(1,55)]
TempLabels = [f"{i:.2f}" for i in TempValues]

#%% Dealing with EnergyTeMPerature RelationShip
plt.cla()
plt.figure(figsize=(3,2))
PrefMeanNrg= np.nanmean(PrefEnergyArray,axis=1) 
ThreshMeanNrg= np.nanmean(ThreshEnergyArray,axis=1) 
for i in range(15):
   plt.plot(TempValues , PrefEnergyArray[:,i],alpha=0.5,color='tab:cyan')#,s=10)
   plt.plot(TempValues , ThreshEnergyArray[:,i],alpha=0.5,color='tab:orange')#,s=10)
#     stats.probplot(ThreshBigArray[i,:], dist="norm", plot=pylab)
plt.plot(TempValues ,PrefMeanNrg,color='b',label="Prefference")  
plt.plot(TempValues, ThreshMeanNrg,color='r',label="Threshold") 
plt.xlabel("Temperature")
plt.ylabel("Energy")

x_coord = 0.57 # x value you want to highlight
y_coord = ThreshMeanNrg[37] # y value you want to highlight

x_highlighting = [0, x_coord, x_coord]
y_highlighting = [y_coord, y_coord, -1]

plt.plot(x_highlighting, y_highlighting,linestyle='--',linewidth=1)
#plt.yticks([])
plt.xlim((0.015,0.8))#, kwargs)
plt.ylim((-1,-0.28))

x_coord = TempValues[18] # x value you want to highlight
y_coord = PrefMeanNrg[18] # y value you want to highlight

x_highlighting = [0, x_coord, x_coord]
y_highlighting = [y_coord, y_coord, -1]

plt.plot(x_highlighting, y_highlighting,linestyle='--',linewidth=1)

plt.xlim((0,0.8))#, kwargs)
plt.ylim((-1,-0.22))
plt.legend()

plt.title("Comparison of Energy Temperature Relationships")
#%% Finding our two dists For our variable
from scipy.stats import norm,lognorm
ThreshDistAtTc = ThreshBigArray[37,:] #SHOUld BE 37
PrefDistAtTc = PrefBigArray[18,:]

# RealWorldData = np.load("PopNumbers.npy")
# ModelX =np.linspace(0,1,256)
# plt.plot(ModelX,ThreshDistAtTc)
# plt.plot(ModelX,PrefDistAtTc)
# RealX=np.linspace(0, 1,len(RealWorldData))
# plt.plot(RealX,RealWorldData)

RealWorldData = np.load("PopNumbers.npy").astype(int)
def Normaliser(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())
def ZNormaliser(arr):
    return (arr - arr.mean()) / arr.std()
NormPrefDistAtTc = Normaliser(PrefDistAtTc)
NormThreshDistAtTc = Normaliser(ThreshDistAtTc)
NormRealWorldData = Normaliser(RealWorldData)

plt.hist(NormThreshDistAtTc,density=True,color='g',alpha=0.4)
plt.hist(NormPrefDistAtTc,density=True,color='r',alpha=0.4)
plt.hist(NormRealWorldData[:5000],bins=20,density=True,alpha=0.6)
#%%
from scipy import stats
N=3
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0.1,0.9,N)))
ThreshDistAtTc = ThreshBigArray[37,:] #SHOUld BE 37
PrefDistAtTc = PrefBigArray[18,:]

ThreeDists ={"Thresh":ThreshDistAtTc
             ,"Pref":PrefDistAtTc
             ,"Real":RealWorldData}
for i in ThreeDists:
    OtherArray = Normaliser(ThreeDists[i])
    n,x = np.histogram(OtherArray, bins=50, density=True)
        #plt.title("Language distribution for")
    density = stats.gaussian_kde(OtherArray)
    y =density(x)
    plt.plot(x,y,label=i)#density(x))
plt.legend()

#%%Power Binning

num_data_points = 6520

# Define the number of bins
num_bins = 20

# Generate logarithmically spaced bin edges
bin_edges = np.logspace(np.log10(1), np.log10(num_data_points), num_bins + 1, base=10.0)


bin_indices = np.digitize(RealWorldData , bin_edges) - 2
bin_totals = np.zeros(num_bins)
for i in range(0,len(RealWorldData)):
    bin_totals[bin_indices[i]] += RealWorldData[i]
#print(bin_indices)
plt.plot(bin_totals)#,bins=40)
#%% Each real world language is only 0.0392638036809816 sim langs#
#Aka 25 irl,
# 5420 real speakers per model one
ScaledPopulations = RealWorldData/(np.sum(RealWorldData,dtype=np.int64)/np.sum(ThreshDistAtTc))

def PopNormaliser(arr,arr2=ThreshDistAtTc):
    A = arr2.min()
    B = arr2.max()
    ans= (arr - arr.min()) / (arr.max() - arr.min())
    return A+ans*(B-A)

plt.plot(np.sort(np.random.choice(PopNormaliser(RealWorldData),size=256))[::-1])
plt.plot(ThreshDistAtTc)
#%%
    
for i in range(10):
    ScaledLangs=PopNormaliser(np.random.choice(ScaledPopulations,size=256))
    
    n,x = np.histogram(ScaledLangs, bins=50, density=True)
        #plt.title("Language distribution for")
    density = stats.gaussian_kde(ScaledLangs)
    y = density(x)
    plt.plot(x,y,alpha=0.3)
    
TwoDists ={"Thresh":ThreshDistAtTc
              ,"Pref":PrefDistAtTc}  
for i in TwoDists:
    OtherArray = PopNormaliser((TwoDists[i]))
    n,x = np.histogram(OtherArray, bins=50, density=True)
        #plt.title("Language distribution for")
    density = stats.gaussian_kde(OtherArray)
    y =density(x)
    plt.plot(x,y,label=i)#density(x))
plt.legend()


#plt.plot(ThreshDistAtTc)
#%%
# Number of data points
num_data_points = 6520

# Number of bins
num_bins = 256

# Generate logarithmically spaced bin edges
bin_edges = np.logspace(np.log10(1), np.log10(num_data_points), num_bins + 1, base=10.0)
RevSortRealWorldData = np.sort(RealWorldData)#[::-1]
# Assign each data point to its corresponding bin
bin_indices = np.digitize(RevSortRealWorldData , bin_edges) - 1

# Verify the number of points in each bin
counts_per_bin = np.bincount(bin_indices)
bin_totals = np.zeros(num_bins)
for i in range(0,len(RealWorldData)):
    bin_totals[bin_indices[256-i]-1] += RevSortRealWorldData[i]
#print(bin_indices)
plt.plot(bin_totals)
# Print the counts per bin
#print(counts_per_bin)

#%%
import numpy as np
import matplotlib.pyplot as plt

# Number of data points
num_data_points = 6520

# Number of bins
num_bins = 256

# Generate logarithmically spaced bin edges
bin_edges = np.logspace(np.log10(1), np.log10(num_data_points), num_bins + 1, base=10.0)

# Verify the number of points in each bin
counts_per_bin = np.bincount(bin_indices)

# Plot the distribution of bin counts against logarithmically spaced bin edges
plt.figure(figsize=(10, 5))
plt.bar(np.log10(bin_edges[:-1]), counts_per_bin[:-1], width=0.1, align='edge', edgecolor='black')
plt.xlabel('Logarithmically spaced bin edges')
plt.ylabel('Number of data points')
plt.title('Distribution of data points in logarithmically spaced bins')
plt.grid(True)
plt.show()

#%%
num_data_points = len(RevSortRealWorldData)
num_bins = 256
based = 2
Upper = np.emath.logn(based,num_data_points)
bin_edges = np.logspace(0, Upper, num_bins + 1, base=based)

bin_edges = np.linspace(0, num_data_points,num_bins+1)
# Assign each data point to its corresponding bin
bin_indices = np.digitize(RevSortRealWorldData , bin_edges) - 1
counts_per_bin = np.bincount(bin_indices)
print(counts_per_bin)
#plt.hist(RevSortRealWorldData,bins=bin_edges)
plt.figure()
plt.hist(counts_per_bin)
