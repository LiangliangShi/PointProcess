# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:52:47 2019

@author: Hasee
"""

from HawkesSimulation1D import HawkesSimulation1D
import tensorflow as tf
import numpy as np
import random

def caculateEG(seqs,beta=0.8):
    global Tf
    Tf=seqs[-1]+random.uniform(0,1)*(seqs[-1]-seqs[-2])
    dex=np.zeros(len(seqs))
    for i in range(1,len(seqs)):
        for j in range(i):
            dex[i]+=np.exp(-beta*(seqs[i]-seqs[j]))
    G_Tj=np.zeros(len(seqs))
    for i in range(len(seqs)):
        G_Tj[i]=(1-np.exp(-beta*(Tf-seqs[i])))/beta
    G=np.sum(G_Tj)
    return dex,G

class BPPP1D(object):
    def __init__(self, n_seq):
        with tf.name_scope('inputs'):
            self.ex=tf.placeholder(tf.float32,[None, n_seq])
            self.Gx=tf.placeholder(tf.float32,[None,1])
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
        self.mu=self.mu_Variable([1])
        self.alpha=self.alpha_Variable([1])
        self.logliklihood=tf.reduce_sum(tf.log(self.ex*self.alpha+self.mu))-self.mu*Tf-self.alpha*self.Gx
           

if __name__ == '__main__':
    mu=1.2
    alpha=0.6
    beta=0.8
    T=1000
    [seqs,lams]=HawkesSimulation1D(mu,alpha,beta,T)
    dex,G=caculateEG(seqs)
    n=len(seqs)
    model=BPPP1D(n)
    sess = tf.Session()
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)        
    for i in range(30000):
        sess.run(model.train_opt,feed_dict={model.ex:dex.reshape([1,n]), model.Gx:G.reshape([1,1])})
        if i%1000==1:
            print('mu:',sess.run(model.mu),'alpha:',sess.run(model.alpha)) 