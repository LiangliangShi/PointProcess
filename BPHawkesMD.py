# -*- coding: utf-8 -*-
"""
Created on Thu May 16 00:14:08 2019

@author: Hasee
"""

from HawkesSimulationMD import HawkesSimulationMD
import tensorflow as tf
import numpy as np
import random

def caculateEGM(seqs,d,beta=1):
    n=len(seqs)
    dex=np.zeros([n,d,d])
    for i in range(1,len(seqs)):
        for j in range(i):
            dex[i,dim[i],dim[j]]+=np.exp(-beta*(seqs[i]-seqs[j]))

    dimmu=np.zeros([n,d])
    for i in range(n):
        dimmu[i,dim[i]]=1.0

    Gddj=np.zeros([d,d])
    for i in range(d):
        for j in range(n):
            Gddj[i,dim[j]]+=(1-np.exp(-beta*(Tf-seqs[j])))/beta
            
    return dex,dimmu,Gddj


class BPPPMD(object):
    def __init__(self, n_seq,d):
        self.n = n_seq
        self.d = d
        self.d2 = d*d
        with tf.name_scope('inputs'):
            self.ex=tf.placeholder(tf.float32,[None, n, d, d])
            self.Gx=tf.placeholder(tf.float32,[None, d, d])
            self.mux=tf.placeholder(tf.float32,[None, n, d])
        with tf.variable_scope('layer'):
            self.addlayer()
        with tf.name_scope('train'):
            loss=-tf.reduce_mean(self.logliklihood)
            self.train_opt=tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    def mu_Variable(self,shape):
        initial = tf.truncated_normal(shape, mean=0.5, stddev=0.2)
        return tf.Variable(initial)

    def alpha_Variable(self,shape):
        initial = tf.truncated_normal(shape, mean=0.5, stddev=0.2)
        return tf.Variable(initial)
    
    
    def addlayer(self):
        exx=tf.reshape(self.ex, [-1,self.d2])
        Gxx=tf.reshape(self.Gx,[-1,self.d2])
        muxx=tf.reshape(self.mux,[-1,self.d])

        self.mu=self.mu_Variable([self.d,1])
        self.alpha=self.alpha_Variable([self.d2,1])

        aa=tf.reshape(tf.matmul(exx,self.alpha),[-1,self.n])
        ma=tf.reshape(tf.matmul(muxx,self.mu),[-1,self.n])
        self.logliklihood=tf.reduce_sum(tf.log(aa+ma))-tf.reduce_sum(self.mu)*Tf-tf.matmul(Gxx,self.alpha)

if __name__ == '__main__':
    mu = np.array([0.1,0.5])
    alpha = np.array([0.2,0.5,0.1,0.3]).reshape([2,2])
    beta = 1
    T = 2000
    [lams,seqs,dim] = HawkesSimulationMD(mu,alpha,beta,T)
    global Tf
    Tf = seqs[-1]+random.uniform(0,1)*(seqs[-1]-seqs[-2])
    n = len(seqs)
    d = len(mu)
    model = BPPPMD(n,d)
    dex,dimmu,Gddj=caculateEGM(seqs,d)
    ex1=dex.reshape([1,n,d,d])
    Gx1=Gddj.reshape([1,d,d])
    mux1=dimmu.reshape([1,n,d])
    sess = tf.Session()
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(100000):
        sess.run(model.train_opt,feed_dict={model.ex:ex1, model.Gx:Gx1, model.mux:mux1})
        if i%1000==1:
            print('mu:',sess.run(model.mu))
            print('alpha:',sess.run(model.alpha))