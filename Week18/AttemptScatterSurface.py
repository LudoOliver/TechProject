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

# Generate some example data (you can replace this with your actual data)
num_points = 55
x_values = np.random.rand(num_points) * 10
y_values = np.random.rand(num_points) * 10
z_values = np.sin(x_values) + np.cos(y_values)



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
        
x_values = CoOrdMatrix[0,1:]
y_values = CoOrdMatrix[1,1:]
z_values = CoOrdMatrix[2,1:]
# Create a 3D surface plot
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_trisurf(x_values, y_values, z_values, cmap='viridis')
ax.scatter(x_values ,z_values ,y_values)# s=200, label='True Position')
# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
#%%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def surface_plot(X,Y,Z,**kwargs):
    """ WRITE DOCUMENTATION
    """
    xlabel, ylabel, zlabel, title = kwargs.get('xlabel',""), kwargs.get('ylabel',""), kwargs.get('zlabel',""), kwargs.get('title',"")
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,Z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()
    plt.close()


surface_plot(x_values, y_values, z_values)#, kwargs)




