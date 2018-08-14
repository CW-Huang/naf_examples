#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 18:25:35 2018

@author: chin-weihuang
"""




import naf

distr = naf.distributions.SwissRoll(0.5)
res = 200
rng = [(-5,5),(-5,5)]


#distr = naf.distributions.FourDiamond()
#res = 300
#rng = [(-10,10),(-10,10)]


### DSF
denaf = naf.DensityEstimator(flowtype=1)
denaf.fit(distr, 2000)
fig = naf.visualize2D(distr, denaf, res=res, rng=rng)

### IAF
denaf = naf.DensityEstimator(flowtype=0)
denaf.fit(distr, 2000)
fig = naf.visualize2D(distr, denaf, res=res, rng=rng)
















