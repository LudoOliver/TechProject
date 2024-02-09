# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:29:51 2024

@author: xd21736
"""
import numpy as np
# x = np.zeros([2,4])+2
# y = np.zeros([3,5])+4

# Combo =[x,y]
# for i in range(len(Combo)):
#     name = "Result"+str(i)+".npz"
#     np.savez(name,Combo[i],allow_pickle=True)


Testa = np.load('Week14TestOut.npz')

FatMatrix = Testa['0']
