# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:46:07 2019

@author: Hasee
"""
import random
import numpy as np
from numpy import *
random.seed(11)
  
            
    
    
def HawkesSimulationMD(mu,alpha,beta,T):
    
    def poisson(slambd):
        return -1/slambd*np.log(random.uniform(0,1))
    
    def IntensitymHawkes(mu,alpha,beta,s,seq,dim):
        result=1.0*mu
        for i in range(len(mu)):
            for j in range(len(seq)):
                if s>=seq[j]:
                    result[i]+=alpha[i,dim[j]]*np.exp(-beta*(s-seq[j]))
    #    print("result:"+str(result))
    #    print(mu)
        return result

    
    #firt event
    lam=1.0*mu
#    cumlam=np.cumsum(lam)
    s=0
#    n=len(mu.tolist())
#    poisson(mu[1])
    lams=[]
    seq=[]
    dim=[]
    
    slam=np.sum(lam)
    s=s+poisson(slam)
    if s>T:
        print('wrong')
    else:
        k=0
        D=random.uniform(0,1)
        while D*slam>np.sum(lam[:k+1]):
            k+=1
        seq.append(s)
        dim.append(k)
        
#        l=cumlam/slam
#        choicei(l)
    
    lams.append(mu)
    tf=0

    while True:
        lam=IntensitymHawkes(mu,alpha,beta,s,seq,dim)
        slam=np.sum(lam)
        if tf==1:
            lams.append(lam)
        # new event
        s=s+poisson(slam)
        if s>T:
            break
        # Attribution
        D=random.uniform(0,1)
        temp=IntensitymHawkes(mu,alpha,beta,s,seq,dim)
        if D<=np.sum(temp)/slam:
            k=0
            
            while D*slam>np.sum(temp[:k+1]):
                k+=1
            seq.append(s)
            dim.append(k)
            

            tf=1
        else:
            tf=-1
            continue
    return lams,seq,dim

if __name__ =='__main__':
    mu=np.array([0.1,0.5])
    alpha=np.array([0.2,0.5,0.1,0.1]).reshape([2,2])
    beta=1
    T=400
    [lams,seq,dim]=HawkesSimulationMD(mu,alpha,beta,T)
    
    mu=np.array([1.2])
    alpha=np.array([0.6]).reshape([1,1])
    beta=0.8
    T=1000
    
    
    
    
    
