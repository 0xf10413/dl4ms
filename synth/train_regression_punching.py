#!/usr/bin/env python3
import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from Network import Network
from AdamTrainer import AdamTrainer
from network import create_core, create_regressor

rng = np.random.RandomState(23455)

data = np.load('../data/processed/data_hdm05.npz')
data_punching = np.hstack([np.arange(259, 303), np.arange(930, 978), np.arange(1650,1703), np.arange(2243,2290), np.arange(2851,2895)])
rng.shuffle(data_punching)

punching_train = data_punching[:len(data_punching)//2]
punching_valid = data_punching[len(data_punching)//2:]

X = data['clips'][punching_train]
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

hands = np.array([60,61,62,48,49,50])

Y = X[:,hands]

I = np.arange(len(X))
rng.shuffle(I)
X, Y = X[I], Y[I]

batchsize = 1
window = X.shape[2]

network_second = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
network_second.load(np.load('network_core.npz'))

network_first = create_regressor(batchsize=batchsize, window=window, input=Y.shape[1])
network = Network(network_first, network_second[1], params=network_first.params)

E = theano.shared(X, borrow=True)
F = theano.shared(Y, borrow=True)

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
trainer.train(network, F, E, filename='network_regression_punch.npz')
