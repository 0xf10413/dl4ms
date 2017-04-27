#!/usr/bin/env python3
import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from network import create_core
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

rng = np.random.RandomState(413342)

X = np.load('../data/processed/data_cmu.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

batchsize = 1
window = X.shape[2]

X = theano.shared(X, borrow=True)

network = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
network.load(np.load('network_core.npz'))

from AnimationPlot import animation_plot

for _ in range(10):

    index = rng.randint(X.shape[0].eval())
    Xorgi = np.array(X[index:index+1].eval())
    Xnois = Xorgi.copy()
    Xnois[:,16*3-1:17*3] = 0.0
    Xrecn = np.array(network(Xnois).eval())    

    Xorgi = (Xorgi * preprocess['Xstd']) + preprocess['Xmean']
    Xnois = (Xnois * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']

    Xrecn = constrain(Xrecn, network[0], network[1], preprocess, multiconstraint(
        foot_sliding(Xorgi[:,-4:].copy()),
        joint_lengths(),
        trajectory(Xorgi[:,-7:-4])), alpha=0.01, iterations=50)

    Xrecn[:,-7:-4] = Xorgi[:,-7:-4]
        
    animation_plot([Xnois, Xrecn, Xorgi], interval=15.15)
        
