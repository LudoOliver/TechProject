# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 12:48:04 2023

@author: farka
"""
import torch as t
import matplotlib.pyplot as plt
a = 0
b = 0
c = 0
for i in range(1,10000):
    if i%10==0:
        a +=1
    if (i*i)%10==0:
        b +=1
    if i*(i-1)%10==0:
        c +=1
print(a)
print(b)
print(c)