#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:43:23 2018

@author: chin-weihuang
"""

import numpy as np
import matplotlib.pyplot as plt
from DensityEstimator import DensityEstimator
from torch.autograd import Variable
import torch


def visualize2D(distr, mdl, res=200, rng=[(-5,5),(-5,5)],
                sample_from_distr=True):
    
    fig = plt.figure(figsize=(8,6))
    if not sample_from_distr:
        pass
    else:
        ax = fig.add_subplot(1,2,1)
        spl = distr.sampler(res**2).numpy()
        xx = spl[:,0]
        yy = spl[:,1]
        ax.hist2d(xx,yy,100, range=rng)
        ax.axis('off')
    
    if isinstance(mdl, DensityEstimator):
        ax = fig.add_subplot(1,2,2)
        x = np.linspace(rng[0][0], rng[0][1], res)
        y = np.linspace(rng[1][0], rng[1][1], res)
        xx,yy = np.meshgrid(x,y)
        X = np.concatenate((xx.reshape(res**2,1),yy.reshape(res**2,1)),1)
        X = X.astype('float32')
        X = Variable(torch.from_numpy(X))
        
        
        context = Variable(torch.FloatTensor(res**2, 1).zero_()) + 2.0
        lgd = Variable(torch.FloatTensor(res**2).zero_())
        zeros = Variable(torch.FloatTensor(res**2, 2).zero_())
                
        
        
        Z = mdl.density(X, lgd, context, zeros).data.numpy().reshape(res,res)
        ax.pcolormesh(xx,yy,np.exp(Z))
        ax.axis('off')
        plt.xlim(rng[0])
        plt.ylim(rng[1])
    
    return fig
        