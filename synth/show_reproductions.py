#!/usr/bin/env python3
import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from network import create_core

rng = np.random.RandomState(23455)

X = np.load('../data/processed/data_edin_locomotion.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

batchsize = 1
window = X.shape[2]

X = theano.shared(X, borrow=True)

network = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
network.load(np.load('network_core.npz'))

from AnimationPlot import animation_plot

for i in range(5):

    index = rng.randint(X.shape.eval()[0]-1)
    Xorgi = np.array(X[index:index+1].eval())
    Xrecn = np.array(network(X[index:index+1]).eval())    

    Xorgi = (Xorgi * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    
    animation_plot([Xorgi, Xrecn], interval=15.15)
    
