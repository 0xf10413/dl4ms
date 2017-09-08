#!/usr/bin/env python3
"""
petit test de dimensionnalité
"""

import os
import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from AdamTrainer import AdamTrainer
from AnimationPlot import animation_plot
from network import create_core
from BiasLayer import BiasLayer
from Conv1DLayer import Conv1DLayer
from ActivationLayer import ActivationLayer
from DropoutLayer import DropoutLayer
from Pool1DLayer import Pool1DLayer
from Depool1DLayer import Depool1DLayer
from Network import Network

rng = np.random.RandomState(23456)

#Xcmu = np.load('../data/processed/data_cmu.npz')['clips']
#Xhdm05 = np.load('../data/processed/data_hdm05.npz')['clips']
#Xmhad = np.load('../data/processed/data_mhad.npz')['clips']
#Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
#Xedin_locomotion = np.load('../data/processed/data_edin_locomotion.npz')['clips']
#Xedin_xsens = np.load('../data/processed/data_edin_xsens.npz')['clips']
#Xedin_misc = np.load('../data/processed/data_edin_misc.npz')['clips']
Xedin_punching = np.load('../data/processed/data_edin_punching.npz')['clips']

#X = np.concatenate([Xhdm05, Xmhad, Xstyletransfer, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)
#X = np.concatenate([Xcmu, Xhdm05, Xmhad, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)
X = Xedin_punching
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

feet = np.array([12,13,14,15,16,17,24,25,26,27,28,29])

Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
Xmean[:,-7:-4] = 0.0
Xmean[:,-4:]   = 0.5

Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)
Xstd[:,feet]  = 0.9 * Xstd[:,feet]
Xstd[:,-7:-5] = 0.9 * X[:,-7:-5].std()
Xstd[:,-5:-4] = 0.9 * X[:,-5:-4].std()
Xstd[:,-4:]   = 0.5

print("Saving…")
np.savez_compressed('preprocess_core_flo.npz', Xmean=Xmean, Xstd=Xstd)

X = (X - Xmean) / Xstd

I = np.arange(len(X))
rng.shuffle(I); X = X[I]

print(X.shape)

E = theano.shared(X[:1,...], borrow=True)
window = 240
dropout = .25
rng = np.random
batchsize = 1

print("At the beginning, dim E =", E.eval().shape)
E = DropoutLayer(amount=dropout, rng=rng)(E)
print("After dropout, dim E =", E.eval().shape)
E = Conv1DLayer(filter_shape=(256, 73, 25),
        input_shape=(batchsize, 73, window),
        rng=rng)(E)
print("After Conv1D, dim E =", E.eval().shape)
E = BiasLayer(shape=(256, 1))(E)
print("After Bias, dim E =", E.eval().shape)
E = ActivationLayer()(E)
print("After Activation, dim E =", E.eval().shape)
E = Pool1DLayer(input_shape=(batchsize, 256, window))(E)
print("And now, for decoder")
E = Depool1DLayer(output_shape=(batchsize, 256, window),
        depooler='random', rng=rng)(E)
print("After Depool1D, dim E =", E.eval().shape)
E = DropoutLayer(amount=dropout, rng=rng)(E)
print("After Dropout, dim E =", E.eval().shape)
E = Conv1DLayer(filter_shape=(73, 256, 25),
        input_shape=(batchsize, 256, window), rng=rng)(E)
print("After Conv1D, dim E =", E.eval().shape)
E = BiasLayer(shape=(73, 1))(E)
print("After Bias, dim E =", E.eval().shape)
