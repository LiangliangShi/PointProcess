# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:58:27 2019

@author: Hasee
"""

import random
import numpy as np
from numpy import *
from HawkesSimulationMD import HawkesSimulationMD
from qqplot1D import qqplot1D
random.seed(11)


def qqplotMD(lams,seq,dim):
    LAM=[]
    dLAM=[]
        
    for i,t1 in enumerate(seq):
        u=dim[i]
        LAMu=0
        if i==0:
            LAMu=mu[u]*t1
    #        print('i='+str(i)+';LAMu='+str(LAMu))
            
        elif u in dim[:i]:
            d0=[i for i,a in enumerate(dim[:i]) if a==u][-1]
            t0=seq[d0]
            LAMu=mu[u]*(t1-t0)
            LAMu+=np.sum([alpha[u,dim[s]]/beta*(np.exp(-beta*(t0-seq[s]))-np.exp(-beta*(t1-seq[s]))) for s in range(d0)])

            LAMu+=np.sum([alpha[u,dim[s]]/beta*(1-np.exp(-beta*(t1-seq[s]))) for s in range(d0,i)])
        else:
            LAMu=mu[u]*(t1)+np.sum([alpha[u,dim[s]]/beta*(1-np.exp(-beta*(t1-seq[s]))) for s in range(i)])
    
        LAM.append(LAMu)
        dLAM.append(u)
        
    LLAM=[]
    for d in range(len(mu)):
        temp=[LAM[i] for i,x in enumerate(dim) if x==d]
        temp.sort()
        qqplot1D(temp,'%dD simulation data'%(d+1))
        LLAM.append(temp)
    return LLAM

if __name__ =='__main__':
    mu=np.array([0.1,0.5])
    alpha=np.array([0.2,0.5,0.1,0.1]).reshape([2,2])
    beta=1
    T=800
    
    [lams,seq,dim]=HawkesSimulationMD(mu,alpha,beta,T)
    qqplotMD(lams,seq,dim)

    