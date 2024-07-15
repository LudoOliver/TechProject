# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:57:15 2024

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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


ThreshTempValues = [0.015*i for i in range(1,55)]
ThreshTempLabels = [f"{i:.2f}" for i in ThreshTempValues]
TempRange = ThreshTempLabels

ThreshBigArray = np.load("ThreshBigArrayFinal.npy")
ThreshEnergyArray = np.load("ThreshEnergyFinal.npy")
#BigXYZArray=np.empty([54*21,3])
CoOrdMatrix = np.zeros([3,1])
for i in range(len(ThreshTempValues)): #calling i x
    CurrentArray = ThreshBigArray[i,:]     
    n,x = np.histogram(CurrentArray, bins=20, density=True)
    #n,x,_ = plt.hist(CriticalArray, bins=20, density=True)
    density = stats.gaussian_kde(CurrentArray)
    y = density(x)
    z = np.ones(np.size(y))*ThreshTempValues[i]
    CurrentPoints = np.vstack([x,y,z])
    CoOrdMatrix = np.hstack([CoOrdMatrix,CurrentPoints])
np.save("ThreshCords",CoOrdMatrix)      
x = CoOrdMatrix[0,1:]
z = CoOrdMatrix[1,1:]
y = CoOrdMatrix[2,1:]
# Create a 3D surface plot
#%%
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.interpolate import griddata

xv = np.linspace(np.min(x), np.max(x), 20)
yv = np.linspace(np.min(y), np.max(y), 20)
[X,Y] = np.meshgrid(xv, yv)
Z = griddata((x,y),z,(X,Y),method='linear')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, 
                       linewidth=0.6, antialiased=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
#fig.colorbar(surf, shrink=0.6)
plt.show()
#%%
# for angle in range(0, 360, 18):
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     ax.view_init(30, angle)
#     surf = ax.plot_surface(X, Y, Z, 
#                            linewidth=0.6, antialiased=True)
   

# plt.show()
num_angles =20
# Create a separate figure for each angle
for angle in range(0, 360, 360 // num_angles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.view_init(30, angle)
    plt.show()
#%%
for angle in range(0, 360, 360 // num_angles):
    for elev in range(-10, 30, 10):  # Elevations from 30 to 50 degrees
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        plt.title(f"View ={angle},Elev = {elev}")#, kwargs)
        ax.view_init(elev, angle)
        plt.show()
# fig = plt.figure()
#%%
num_angles=4
for angle in range(0, 360, 360 // num_angles):
    for elev in range(-10, 110, 20):  # Elevations from 30 to 50 degrees
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        plt.title(f"View ={angle},Elev = {elev}")#, kwargs)
        ax.view_init(elev, angle)
        plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #ax.plot_trisurf(x_values, y_values, z_values, cmap=)
# ax.scatter(x_values ,z_values ,y_values)# s=200, label='True Position')
# # Set labels
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # Show the plot
# plt.show()
# #%%
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

# def surface_plot(X,Y,Z,**kwargs):
#     """ WRITE DOCUMENTATION
#     """
#     xlabel, ylabel, zlabel, title = kwargs.get('xlabel',""), kwargs.get('ylabel',""), kwargs.get('zlabel',""), kwargs.get('title',"")
#     fig = plt.figure()
#     fig.patch.set_facecolor('white')
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X,Y,Z)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_zlabel(zlabel)
#     ax.set_title(title)
#     plt.show()
#     plt.close()


# surface_plot(x_values, y_values, z_values)#, kwargs)




