#!/usr/bin/env python3
import os
import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from AdamTrainer import AdamTrainer
from AnimationPlot import animation_plot
from network import create_core, create_core_flo

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
#X = np.concatenate([Xcmu, Xhdm05, Xmhad, Xstyletransfer, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)
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

print("Saving preprocessing data (mean & std)…")
np.savez_compressed('preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

X = (X - Xmean) / Xstd

I = np.arange(len(X))
rng.shuffle(I); X = X[I]

print(X.shape)

E = theano.shared(X, borrow=True)

batchsize = 1
network = create_core_flo(rng=rng, batchsize=batchsize, window=X.shape[2])

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
print("Training…")
trainer.train(network, E, E, filename='network_core_flo.npz')
