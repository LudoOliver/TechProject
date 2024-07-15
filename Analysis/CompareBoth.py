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
#ThreshDistAtTc = ThreshBigArray[37,:]

ThreshLong = np.load("ThreshDistMatrixLong.npy")
IndicesICareFor = (17,27,37,38)
ThreshTcNew = ThreshLong[3,:]
ThreshTCNew = ThreshTcNew[np.where(ThreshTcNew>0)]



PrefLong = np.load("PrefDistMatrixLong.npy")
PrefIndicesICareFor = (17,22,23,24)
PrefTcNew = PrefLong[3,:]
PrefTcNew = PrefTcNew[np.where(PrefTcNew>0)]

ThreshDistAtTc = ThreshTCNew

plt.figure(figsize=(4.5,3),dpi=300)
plt.hist(ThreshDistAtTc , bins=40, density=True, alpha=0.4,edgecolor='black', linewidth=0.5)# color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
mu, std = norm.fit(ThreshDistAtTc)
realparam = [.1, 0, np.exp(10)]
param = lognorm.fit(ThreshDistAtTc)
pdf_fitted = lognorm.pdf(
    x, *param[:-2], loc=param[-2], scale=param[-1])
plt.plot(x, pdf_fitted,'r' ,label="Log-Normal",linewidth=2,linestyle='--')

p = norm.pdf(x, mu, std)
plt.plot(x, p, 'b', label="Normal",linewidth=2,linestyle='-.')
DistName = "ThreshDistForLogLog.jpeg"
#plt.legend()
#plt.yscale("log")
plt.xticks([],[])
plt.yticks([],[])
plt.title("Threshold Language Distribution at $T_C$")
plt.xlabel("N speakers")
plt.ylabel("$n_s$ number of languages")

plt.legend()
plt.show()
from scipy.stats import goodness_of_fit
NRmlScore = goodness_of_fit(norm, ThreshDistAtTc).pvalue
LogNRmlScore = goodness_of_fit(lognorm, ThreshDistAtTc).pvalue
print("Thresh")
print(LogNRmlScore,'log')
print(NRmlScore,'nrm')
#%%
from scipy.stats import norm,lognorm
PrefLong = np.load("PrefDistMatrixLong.npy")
PrefIndicesICareFor = (17,22,23,24)
PrefTcNew = PrefLong[0,:]
PrefTcNew = PrefTcNew[np.where(PrefTcNew>0)]
PrefDistAtTc = PrefTcNew #PrefBigArray[18,:]

plt.figure(figsize=(4.5,3),dpi=300)
plt.hist(PrefDistAtTc , bins=40, density=True, alpha=0.4,edgecolor='black', linewidth=0.5)# color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
mu, std = norm.fit(PrefDistAtTc)
realparam = [.1, 0, np.exp(10)]
param = lognorm.fit(PrefDistAtTc)
pdf_fitted = lognorm.pdf(
    x, *param[:-2], loc=param[-2], scale=param[-1])
plt.plot(x, pdf_fitted,'r' ,label="Log-Normal",linewidth=2,linestyle='--')

p = norm.pdf(x, mu, std)
plt.plot(x, p, 'b', label="Normal",linewidth=2,linestyle='-.')
DistName = "ThreshDistForLogLog.jpeg"
#plt.legend()
plt.xticks([],[])
plt.yticks([],[])
plt.title("Preference Language Distribution at $T=0.27$")
plt.xlabel("N speakers")
plt.ylabel("$n_s$ number of languages")
#plt.yscale("log")
plt.xscale("log")#, kwargs)
plt.legend()
plt.show()
#%%

PrefBigArray = np.load("FPrefDistMatrix.npy")
PrefEnergyArray = np.load("FPrefEnergyMatrix.npy")
ThreshBigArray = np.load("ThreshBigArrayFinal.npy")
ThreshEnergyArray = np.load("ThreshEnergyFinal.npy")


TempValues = [0.015*i for i in range(1,55)]
TempLabels = [f"{i:.2f}" for i in TempValues]

ThreshDistAtTc = ThreshBigArray[37,:] 
PrefDistAtTc = PrefBigArray[18,:]
ThreshDistAtMin =  ThreshBigArray[27,:]
RealWorldData = np.load("PopNumbers.npy").astype(int)
SortedReal = np.sort(RealWorldData)[::-1]
PrefNRmlScore =np.zeros(len(TempValues))
PrefLogNRmlScore= np.zeros(len(TempValues))
ThreshNrmlScore = np.zeros(len(TempValues))
ThreshLogNrmlScore = np.zeros(len(TempValues))
from scipy.stats import goodness_of_fit

for i in range(0,len(TempValues)):
    a = PrefBigArray[i,:]
    a = a[a != 0]
    b = ThreshBigArray[i,:]
    b = b[b != 0]
    PrefNRmlScore[i] = goodness_of_fit(norm, a).pvalue
    ThreshNrmlScore[i] = goodness_of_fit(norm, b).pvalue
    
    PrefLogNRmlScore[i] = goodness_of_fit(norm, np.log10(a) ).pvalue
    ThreshLogNrmlScore[i] = goodness_of_fit(norm, np.log10(b) ).pvalue
    print(i)
#%%
PrefLong = np.load("PrefDistMatrixLong.npy")
PrefIndicesICareFor = (17,22,23,24)
PrefTcNew = PrefLong[3,:]
PrefTcNew = PrefTcNew[np.where(PrefTcNew>0)]
for i in range(0,4):
    a = PrefLong[i,:]
    a = a[a != 0]
    NewPrefNRmlScore = goodness_of_fit(norm, a).pvalue
    lgnrml = goodness_of_fit(norm, np.log10(a)).pvalue
    print(f"at {TempValues[PrefIndicesICareFor[i]]} p normal = {NewPrefNRmlScore}")
    print(f"p log-normal = {lgnrml}")
#%%
import matplotlib
#%%
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.show()
plt.figure(figsize=(4.5,3))
plt.plot((0.36,0.36),(0,1),color='k',linestyle='--',lw="3",label="T=0.36",alpha=0.7)
plt.plot(TempValues,PrefNRmlScore,label="Normal")
plt.plot(TempValues,PrefLogNRmlScore,label="Log-Normal")
plt.title("Preference distribution modes")
plt.xlabel("Temperature")
plt.ylabel("P-value")

plt.legend()
plt.figure(figsize=(4.5,3))
plt.plot((0.57,0.57),(0,1),color='k',linestyle='--',lw="3",label="T=0.57",alpha=0.7)
plt.plot(TempValues,ThreshNrmlScore,label="Normal")
plt.plot(TempValues,ThreshLogNrmlScore,label="Log-Normal")
plt.title("Threshold distribution modes")
plt.xlabel("Temperature")
plt.ylabel("P-value")
plt.legend()
plt.show()
#%%
from scipy.stats import goodness_of_fit
NRmlScore = goodness_of_fit(norm, ThreshDistAtTc).pvalue
LogNRmlScore = goodness_of_fit(lognorm, ThreshDistAtTc).pvalue
print(LogNRmlScore,'log')
print(NRmlScore,'nrm')





# #%%%