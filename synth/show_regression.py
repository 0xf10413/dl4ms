#!/usr/bin/env python3
import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from Network import Network
from network import create_core, create_regressor

rng = np.random.RandomState(23455)

data = np.load('../data/processed/data_edin_locomotion.npz')['clips']

I = np.arange(len(data))
rng.shuffle(I)

data_train = data[I[:len(data)//2]]
data_valid = data[I[len(data)//2:]]

X = data_valid
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']
Y = X[:,-7:]

batchsize = 1

network_first = create_regressor(batchsize=batchsize, window=X.shape[2], input=Y.shape[1], dropout=0.0)
network_second = create_core(batchsize=batchsize, window=X.shape[2], dropout=0.0, depooler=lambda x,**kw: x/2)
network = Network(network_first, network_second[1], params=network_first.params)
network.load(np.load('network_regression.npz'))

from AnimationPlot import animation_plot

for i in range(5):

    index = rng.randint(len(X)-1)
    Xorig = np.array(X[index:index+1])
    Xrecn = np.array(network(Y[index:index+1]).eval())
    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn[:,-7:] = Xorig[:,-7:]
    
    animation_plot([Xorig, Xrecn], interval=15.15)
    
