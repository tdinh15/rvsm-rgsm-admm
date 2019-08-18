# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 11:43:59 2019

@author: I w a y
"""
import torch
import numpy as np
from numpy import linalg as LA


a = {'name':'ken',
     'epsilon':0.1,
     'age':29,}


a = torch.rand(3,4,2,2)

weight = a.clone()
shape = weight.shape
weight2d = weight.reshape(shape[0], -1)
shape2d = weight2d.shape
column_l2_norm = LA.norm(weight2d, 3, axis=0)

#print(weight)
#print(column_l2_norm)


wt = a.clone()
ch_wt = wt.pow(2).sum(dim=[0,2,3]).add(1e-8).pow(1/2.)
print(ch_wt)

percent = 50
for i in range(len(ch_wt)):
    pcen = np.percentile(abs(ch_wt),percent)
    under_threshold = abs(ch_wt) < pcen
    ch_wt[under_threshold] = 0
    wt[:,i,:,:] *= ch_wt[i]
print(ch_wt)
print(wt)


b = [0,0,0,0,0,0,0,1]
np.percentile(b,50)