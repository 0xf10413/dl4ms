#!/usr/bin/env python3
import sys
import time
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from Network import Network
from network import create_core, create_regressor
from constraints import constrain, foot_sliding, joint_lengths, multiconstraint

rng = np.random.RandomState(23455)

data = np.load('../data/processed/data_hdm05.npz')
data_kicking = np.hstack([np.arange(199, 246), np.arange(862, 906), np.arange(1582,1640), np.arange(2188,2233), np.arange(2796,2844)])
rng.shuffle(data_kicking)

kicking_train = data_kicking[:len(data_kicking)//2]
kicking_valid = data_kicking[len(data_kicking)//2:]

X = data['clips'][kicking_valid]
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

feet = np.array([9,10,11,12,13,14,21,22,23,24,25,26])

Y = X[:,feet]

batchsize = 1
window = X.shape[2]

network_first = create_regressor(batchsize=batchsize, window=window, input=Y.shape[1], dropout=0.0)
network_second = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw:x/2)
network_second.load(np.load('network_core.npz'))
network = Network(network_first, network_second[1], params=network_first.params)
network.load(np.load('network_regression_kick.npz'))

from AnimationPlot import animation_plot

for i in range(len(X)):

    network_func = theano.function([], network(Y[i:i+1]))

    Xorig = np.array(X[i:i+1])
    start = time.clock()
    Xrecn = network_func()
    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = constrain(Xrecn, network_second[0], network_second[1], preprocess, multiconstraint(
        foot_sliding(Xrecn[:,-4:].copy()),
        joint_lengths()), alpha=0.01, iterations=50)

    animation_plot([Xorig, Xrecn], interval=15.15)

