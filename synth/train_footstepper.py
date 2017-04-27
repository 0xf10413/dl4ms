#!/usr/bin/env python3
import os
import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')
sys.path.append('../motion')

from ActivationLayer import ActivationLayer
from DropoutLayer import DropoutLayer
from Pool1DLayer import Pool1DLayer
from AdamTrainer import AdamTrainer
from network import create_core, create_regressor, create_footstepper

rng = np.random.RandomState(23455)

data = np.load('../data/processed/data_edin_locomotion.npz')['clips']

I = np.arange(len(data))
rng.shuffle(I)

data_train = data[I[:len(data)//2]]
data_valid = data[I[len(data)//2:]]

X = data_train
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

T = X[:,-7:-4]
F = X[:,-4:]

W = np.zeros((F.shape[0], 5, F.shape[2]))

for i in range(len(F)):

    w = np.zeros(F[i].shape)

    for j in range(F[i].shape[0]):
        last = -1
        for k in range(1, F[i].shape[1]):
            if last == -1 and F[i,j,k-1] < 0 and F[i,j,k-0] > 0: last = k; continue
            if last == -1 and F[i,j,k-1] > 0 and F[i,j,k-0] < 0: last = k; continue
            if F[i,j,k-1] > 0 and F[i,j,k-0] < 0:
                if k-last+1 > 10 and k-last+1 < 60:
                    w[j,last:k+1] = np.pi/(k-last)
                else:
                    w[j,last:k+1] = w[j,last-1]
                last = k
                continue
            if F[i,j,k-1] < 0 and F[i,j,k-0] > 0:
                if k-last+1 > 10 and k-last+1 < 60:
                    w[j,last:k+1] = np.pi/(k-last)
                else:
                    w[j,last:k+1] = w[j,last-1]
                last = k
                continue

    c = np.zeros(F[i].shape)

    for k in range(0, F[i].shape[1]):
        window = slice(max(k-100,0),min(k+100,F[i].shape[1]))
        ratios = (
            np.mean((F[i,:,window]>0).astype(np.float), axis=1) /
            np.mean((F[i,:,window]<0).astype(np.float), axis=1))
        ratios[ratios==np.inf] = 100
        c[:,k] = ((np.pi*ratios) / (1+ratios))

    w[w==0.0] = np.nan_to_num(w[w!=0.0].mean())

    W[i,0:1] = w.mean(axis=0)
    W[i,1:5] = c

    # import matplotlib.pyplot as plt
    # plt.plot(F[i,0])
    # plt.plot(np.sin(np.cumsum(W[i,0:1])))
    # plt.ylim([-1.1, 1.1])
    # plt.show()

print(T.shape, W.shape)

Wmean = W.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
Wstd = W.std(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
W = (W - Wmean) / Wstd

np.savez_compressed('preprocess_footstepper_flo.npz', Wmean=Wmean, Wstd=Wstd)

I = np.arange(len(T))
rng.shuffle(I)
T, F, W = T[I], F[I], W[I]

batchsize = 1

T, W = theano.shared(T), theano.shared(W)

network = create_footstepper(batchsize=batchsize, window=X.shape[2], dropout=0.1)
trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.0001)
trainer.train(network, T, W, filename='network_footstepper_flo.npz')

