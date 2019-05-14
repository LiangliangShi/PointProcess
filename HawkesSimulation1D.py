# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 21:06:21 2019

@author: Hasee
"""

import random
import numpy as np
from numpy import *
random.seed(11)
   
def HawkesSimulation1D(mu=1.2, alpha=0.6 ,beta=0.8, T=100):
    if alpha/beta>1:
        print('wrong')
    seq=[]
    lams=[]
    s=0
    def lama(s,seq):
        #betadeT=[-beta*(s-t) for t in seq if s>=t]
        betadeT = (np.multiply(-beta, s-np.array(seq))).tolist()
        return mu+alpha*np.sum(np.exp(betadeT))
    
    def poisson(lam):
        return -1/lam*np.log(random.uniform(0,1))
    
    lam=mu
    s=s+poisson(lam)#first step
    
    if s<T:
        seq.append(s)
        lams.append(mu)
    else:
        print("wrong")
        
    tf=0
    while True:
        lam=lama(s,seq)
        if tf==1:
            lams.append(lam)
        s=s+poisson(lam)
        if s>T:
            break
        D=random.uniform(0,1)
        if D<=lama(s,seq)/lam:
            seq.append(s)
            tf=1
        else:
            tf=-1
            continue
    
    return [[se for se in seq if se<=T],lams]

if __name__ =='__main__':
    mu=1.2
    alpha=0.6
    beta=0.8
    T=10
    [seqs,lams]=HawkesSimulation1D(mu,alpha,beta,T)
#