#!/usr/bin/env python3
import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
"""
Anime des n-uplets de styletransfer
"""

sys.path.append('../nn')
sys.path.append('../synth')

from network import create_core
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

rng = np.random.RandomState(23455)

print("Loading data")
Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
Xhdm05 = np.load('../data/processed/data_hdm05.npz')['clips']
Xedin_locomotion = np.load('../data/processed/data_edin_locomotion.npz')['clips']
Xedin_misc = np.load('../data/processed/data_edin_misc.npz')['clips']
Xedin_punch = np.load('../data/processed/data_edin_punching.npz')['clips']

Xstyletransfer = np.swapaxes(Xstyletransfer, 1, 2).astype(theano.config.floatX)
Xhdm05 = np.swapaxes(Xhdm05, 1, 2).astype(theano.config.floatX)
Xedin_locomotion = np.swapaxes(Xedin_locomotion, 1, 2).astype(theano.config.floatX)
Xedin_misc = np.swapaxes(Xedin_misc, 1, 2).astype(theano.config.floatX)
Xedin_punch = np.swapaxes(Xedin_punch, 1, 2).astype(theano.config.floatX)

preprocess = np.load('../synth/preprocess_core.npz')


def create_network(batchsize, window):
    network = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2, rng=rng)
    network.load(np.load('../synth/network_core.npz'))
    return network

pairings = [
  (241, 242, 243, 259 ), # forward, paused, forward2, turn
  (241, 233, 264, 272),# walk, kick, punch, running
  (264, 58, 337), # punch neutral, angry, old
  (241, 39, 317), # walk neutral, angry, old
]

print("Done loading data")
from AnimationPlot import animation_plot

for triple in pairings:
    S = [Xstyletransfer[i:i+1] for i in triple]
    for i in range(len(triple)):
        for j in range(i):
            print("norm2 between {} and {} is {}".format(
                triple[i], triple[j], np.mean((S[i]-S[j])**2)
                ))
    print()
    animation_plot(S, interval=15.15)
