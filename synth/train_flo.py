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
from Network import Network
from network import create_core, create_core_flo

from os.path import isfile

rng = np.random.RandomState(23456)

Xcmu = np.load('../data/processed/data_cmu.npz')['clips']
Xhdm05 = np.load('../data/processed/data_hdm05.npz')['clips']
Xmhad = np.load('../data/processed/data_mhad.npz')['clips']
Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
Xedin_locomotion = np.load('../data/processed/data_edin_locomotion.npz')['clips']
Xedin_xsens = np.load('../data/processed/data_edin_xsens.npz')['clips']
Xedin_misc = np.load('../data/processed/data_edin_misc.npz')['clips']
Xedin_punching = np.load('../data/processed/data_edin_punching.npz')['clips']

#X = np.concatenate([Xhdm05, Xmhad, Xstyletransfer, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)
X = np.concatenate([Xcmu, Xhdm05, Xmhad, Xstyletransfer, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)
#X = Xedin_punching
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

feet = np.array([12,13,14,15,16,17,24,25,26,27,28,29])


print("Loading preprocessed data (mean & std)…")
C = np.load('preprocess_core.npz')
Xmean = C['Xmean']
Xstd = C['Xstd']

X = (X - Xmean) / Xstd

I = np.arange(len(X))
rng.shuffle(I); X = X[I]


E = theano.shared(X, borrow=True)

batchsize = 1

print("Training outermost layers...")
network = create_core_flo(rng=rng, batchsize=batchsize, window=X.shape[2])
d1, _d2, _d3, _r3, _r2, r1 = network.layers
trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
N1 = Network(d1, r1)

filename = 'network_core_flo_l1.npz'
trainer.train(N1, E, E, filename=filename)
print()

print("Training middle layers...")
network = create_core_flo(rng=rng, batchsize=batchsize, window=X.shape[2],
        learning_layer=[False, True, True])
d1, d2, d3, r3, r2, r1 = network.layers
N1 = Network(d1, r1)
N1.load(np.load(filename))
d1.freeze(); r1.freeze()

filename = 'network_core_flo_l2.npz'
trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
N2 = Network(d1, d2, r2, r1)
trainer.train(N2, E, E, filename=filename)
print()

print("Training outermost layers...")
network = create_core_flo(rng=rng, batchsize=batchsize, window=X.shape[2],
        learning_layer=[False, False, True])
d1, d2, d3, r3, r2, r1 = network.layers
N2 = Network(d1, d2, r2, r1)
N2.load(np.load(filename))
d1.freeze(); r1.freeze()
d2.freeze(); r2.freeze()

filename = 'network_core_flo_l3.npz'
trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
N3 = network
trainer.train(N3, E, E, filename=filename)
print()

print("Fine tuning everyone...")
network = create_core_flo(rng=rng, batchsize=batchsize, window=X.shape[2],
        learning_layer=[False, False, False])
network.load(np.load(filename))

filename = 'network_core_flo_all.npz'
trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
trainer.train(network, E, E, filename=filename)
print()

