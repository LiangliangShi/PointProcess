# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 23:33:41 2019

@author: Hasee
"""

from HawkesSimulation1D import HawkesSimulation1D

import matplotlib.pyplot as plt  

import numpy as np   ##科学计算库 
import matplotlib.pyplot as plt  ##绘图库

def qqplot1D(Yi,title='QQplot'):

    n=len(Yi)
    s=np.linspace(1,n,n)
    x=(s-0.5)/n
    Xi=-np.log(1-x)

    plt.scatter(Xi,Yi,color="blue",linewidth=0.1) #label="real data",
    
    #画直线
    x=np.linspace(0,Xi[-1],100) ##在0-15直接画100个连续点
    y=1*x+0 ##函数式
    plt.plot(x,y,color="red",linewidth=2) 
    plt.legend(loc='lower right') #绘制图例
    plt.title(title,fontsize='x-large')
    plt.xlabel('quantiles of exponential distribution')
    plt.ylabel('quantiles of input sample')

    plt.show()
    

    


if __name__ =='__main__':

    mu=1.2
    alpha=0.6
    beta=0.8
    T=1000
    [seqs,lams]=HawkesSimulation1D(mu,alpha,beta,T)
    LAM=[]
    LAM.append(mu*seqs[0])
    sq=[]
    for i in range(len(seqs)-1):
        sam=mu*(seqs[i+1]-seqs[i])
        if sq==[]:
            sq.append(seqs[i])
        else:
            betadet=[np.exp(-beta*(seqs[i]-t))-np.exp(-beta*(seqs[i+1]-t)) for t in sq]
            sq.append(seqs[i])
            LAM.append(sam+np.sum(np.array(betadet)*alpha/beta))
    
    LAM.sort()
    qqplot1D(LAM,'QQplot')


