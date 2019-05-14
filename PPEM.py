# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:16:55 2019

@author: Hasee
"""
import numpy as np
from HawkesSimulationMD import HawkesSimulationMD
from HawkesSimulation1D import HawkesSimulation1D


def PPMLE(seq,dim,mu,alpha,beta,T,kstep=100):
    
    p=np.zeros([len(seq),len(seq)])
    for k in range(kstep):
        ####Estep
        for i,sq in enumerate(seq):
            dd=dim[i]
            sumj=0
            if i>0:
                for j in range(i):
                    sumj+=alpha[dd,dim[j]]*np.exp(-beta*(sq-seq[j]))
            
            p[i,i]=mu[dd]/(mu[dd]+sumj)
            for j in range(i):
                p[i,j]=alpha[dd,dim[j]]*np.exp(-beta*(sq-seq[j]))/(mu[dd]+sumj)
        
        ####Mstep
        su=np.zeros(len(mu))
        for i,sq in enumerate(seq):
            su[dim[i]]+=p[i,i]
        
        mu=1.0*su/T
        
        sa=np.zeros([len(mu),len(mu)])
        for i,sq in enumerate(seq):
            for j in range(i):
                sa[dim[i],dim[j]]+=p[i,j]
                
        sG=np.zeros(len(mu))
        for i,sq in enumerate(seq):
            sG[dim[i]]+=1/beta*(1-np.exp(-beta*(T-sq)))
#            print(sG)
        
        for u in range(len(mu)):
            for v in range(len(mu)):
                alpha[u,v]=sa[u,v]/sG[v]
#        print(sa)
#        print(sG)
        if k%20==1:
            print(mu)
            print(alpha)
        
    return mu,alpha

def PPMLE1D(mu,alpha,beta,seqs,T,maxstep=500):
    def Gfun(seqs,T,beta):
        return np.sum([1/beta*(1-np.exp(-beta*(T-sq))) for sq in seqs])
    k=1
    while True:
        if k>maxstep:
            break
        pp=np.zeros([len(seqs),len(seqs)])
        for i,sq in enumerate(seqs):
            if i==0:
                pp[i,i]=1
            else:
                sumj=0
                for j,sj in enumerate(seqs[:i]):
                    sumj+=alpha*np.exp(-beta*(sq-sj))
                pp[i,i]=mu/(mu+sumj)
                for j,sj in enumerate(seqs[:i]):
                    pp[i,j]=alpha*np.exp(-beta*(sq-sj))/(sumj+mu)
#            print(sumj)
                    
        mu1=np.sum(np.diag(pp))/T
        alpha1=(np.sum(np.sum(pp))-np.sum(np.diag(pp)))/Gfun(seqs,T,beta)
        
        
        
#        print('kstep:'+str(k)+'   mu:'+str(mu)+'   alpha:'+str(alpha)+';')
        if np.abs(mu1-mu)/mu<0.001 and np.abs(alpha1-alpha)/alpha<0.001:
            break
        else:
            if k%30==0:
                print('kstep:'+str(k)+'   mu:'+str(mu)+'   alpha:'+str(alpha)+';')
            mu=mu1
            alpha=alpha1
            k+=1
    return mu,alpha

#[mu,alpha]=PPMLE(seq,dim,mu0,alpha0)
#
#
#mu=1.2
#alpha=0.6
#beta=0.8
#T=200
#[seq,lams]=HawkesSimulation1D(mu,alpha,beta,T)
#dim=[0]*len(seq)      
###
###
#mu0=np.array([1.5]).reshape([1,1])
#alpha0=np.array([0.5]).reshape([1,1])
#kstep=50
#[mu,alpha]=PPMLE(seq,dim,mu0,alpha0,beta)
#mu=1*mu0
#alpha=alpha0*1
#[lams,seq,dim]=HawkesSimulationMD(mu0,alpha0,beta,T)
#mu=1.2
#alpha=0.6
#
#
#
#seq0=[sq for i,sq in enumerate(seq) if dim[i]==0]
#seq1=[sq for i,sq in enumerate(seq) if dim[i]==1]
#
#mu0=0.5
#alpha0=0.3
#[mu00,alpha00]=PPMLE1D(mu0,alpha0,beta,seq0,T,maxstep=3)
#[mu01,alpha01]=PPMLE1D(mu0,alpha0,beta,seq1,T,maxstep=3)
#print(mu00,mu01)
#print(alpha00,alpha01)
#mu1=np.array([0.1,0.2])
#alpha1=np.array([0.3,0.1,0.1,0.3]).reshape([2,2])
#mu_1,alpha_1=PPMLE(seq,dim,mu1,alpha1,beta,kstep=1000)
#
#
#simmu=np.array([1.2,1.5])
#simalpha=np.array([0.5,0.3,0.1,0.6]).reshape([2,2])
#beta=1
#T=200
#[lams,seq,dim]=HawkesSimulationMD(simmu,simalpha,beta,T)
#mu1=np.array([1.1,1.2])
#alpha1=np.array([0.3,0.1,0.1,0.3]).reshape([2,2])
#mu_1,alpha_1=PPMLE(seq,dim,mu1,alpha1,beta,kstep=100)
#mu_1=np.array([1.75624749,0.9248431])
#alpha_1=np.array([0.42929134,0.272827,0.10283865,0.70370825]).reshape([2,2])
#from numpy import *;#导入numpy的库函数
#import numpy as np
#print(mat(eye(2)-simalpha/beta).I*mat(simmu).T)
#print(mat(eye(2)-alpha_1/beta).I*mat(mu_1).T)












