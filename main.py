# -*- coding: utf-8 -*-
"""
Created on Tue May 14 19:09:21 2019

@author: Hasee
"""

import numpy as np
from HawkesSimulationMD import HawkesSimulationMD
from HawkesSimulation1D import HawkesSimulation1D
from PPEM import PPMLE
from numpy import *



if __name__ =='__main__':
    ## simulation
    mu=np.array([0.1,0.5])
    alpha=np.array([0.2,0.5,0.1,0.3]).reshape([2,2])
    beta=1
    T=2000
    [lams,seq,dim]=HawkesSimulationMD(mu,alpha,beta,T)
    
    
    ##initial
    mu0=np.array([0.2,0.4])
    alpha0=np.array([0.1,0.2,0.1,0.3]).reshape([2,2])
    
    [pmu,palpha]=PPMLE(seq,dim,mu0,alpha0,beta,T,kstep=90)

    ## average
    seq0=[sq for i,sq in enumerate(seq) if dim[i]==0]
    seq1=[sq for i,sq in enumerate(seq) if dim[i]==1]
    print("average of dims")
    print("Poisson:",len(seq0)/T,len(seq1)/T)
    print("true:",mat(eye(len(mu))-alpha/beta).I*mat(mu).T)
    print("esimation:",mat(eye(len(pmu))-palpha/beta).I*mat(pmu).T)
