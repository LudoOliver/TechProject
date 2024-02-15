# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:20:05 2024

@author: xd21736
"""

from etho import pyetho

x = pyetho.country_list()
y={}
for i in x:
    for j in pyetho.country_languages(country_name=i):
        if j in y:
            y[j].append(i)
        else: 
            y.update({j:[i]})
            
Languages = y.keys()

Pops = dict.fromkeys(Languages)
for i in Languages:
    temp = [0]
    for a in y[i]:
        #Pops[i] 
        temp.append((pyetho.language_population(country_name=a,lang=i)))
    Pops.update({i:temp})