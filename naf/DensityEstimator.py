#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:45:34 2018

@author: chin-weihuang
"""


from torchkit import flows, nn as nn_, utils
from torch import optim, nn
from torch.autograd import Variable
import torch


class DensityEstimator(object):
    
    def __init__(self, flowtype=0, dim=2, dimh=64, 
                 n=64, num_hid_layers=2,
                 act=nn.ELU(), num_flow_layers=2, 
                 num_ds_dim=16, num_ds_layers=1,
                 lr=0.005, betas=(0.9,0.999)):
        if flowtype == 0:
            flow = flows.IAF
            
        elif flowtype == 1:
            flow = lambda **kwargs:flows.IAF_DSF(num_ds_dim=num_ds_dim,
                                                 num_ds_layers=num_ds_layers,
                                                 **kwargs)
            
           
        elif flowtype == 2:
            flow = lambda **kwargs:flows.IAF_DDSF(num_ds_dim=num_ds_dim,
                                                  num_ds_layers=num_ds_layers,
                                                  **kwargs)
            

        sequels = [nn_.SequentialFlow(
            flow(dim=dim,
                 hid_dim=dimh,
                 context_dim=1,
                 num_layers=num_hid_layers+1,
                 activation=act,
                 fixed_order=True),
            flows.FlipFlow(1)) for i in range(num_flow_layers)] + \
                  [flows.LinearFlow(dim, 1),]
        
        self.mdl = nn.Sequential(
                *sequels)
        
        
        
        self.optim = optim.Adam(self.mdl.parameters(), lr=lr, betas=betas)
        self.n = n
        
        self.context = Variable(torch.FloatTensor(n, 1).zero_()) + 2.0
        self.lgd = Variable(torch.FloatTensor(n).zero_())
        self.zeros = Variable(torch.FloatTensor(n, 2).zero_())
        
        
    def density(self, spl, lgd=None, context=None, zeros=None):
        lgd = self.lgd if lgd is None else lgd
        context = self.context if context is None else context
        zeros = self.zeros if zeros is None else zeros
        z, logdet, _ = self.mdl((spl, lgd, context))
        losses = - utils.log_normal(z, zeros, zeros+1.0).sum(1) - logdet
        return - losses

        
    def fit(self, distr, total=2000, verbose=True):
        
        sampler = distr.sampler
        n = self.n
        
        for it in range(total):

            self.optim.zero_grad()
            
            spl = sampler(n)
            
            losses = - self.density(spl)
            
            loss = losses.mean()
            
            loss.backward()
            self.optim.step()
            
            if ((it + 1) % 100) == 0 and verbose:
                print 'Iteration: [%4d/%4d] loss: %.8f' % \
                    (it+1, total, loss.data[0])
            
            