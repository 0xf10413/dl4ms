#!/usr/bin/env python3
import sys

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
import lasagne

from simplegan import SimpleGAN

np.random.seed(43)

####################
## Modèle de GAN ##
####################




######################################################
## Fonction "how_to" de construction intermédiaires ##
######################################################

###############################################
## Fonctions supplémentaires d'aide au build ##
###############################################
def norm_unnorm(database):
    mean_ = np.mean(database, axis=0)
    prepared = database - mean_
    min_ = np.min(prepared, axis=0)
    max_ = np.max(prepared, axis=0)
    scale = max_ - min_
    scale[scale == 0] = 1
    return lambda x: (x - mean_)/scale, lambda y: y*scale + mean_

####################
## Test du script ##
####################

if __name__ == "__main__":
    Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
    Xstyletransfer = np.swapaxes(Xstyletransfer, 1, 2).astype(theano.config.floatX)
    preprocess = np.load('../synth/preprocess_core.npz')
    Xstyletransfer = (Xstyletransfer - preprocess['Xmean']) / preprocess['Xstd']

    pairings = [(i, Xstyletransfer) for i in range(len(Xstyletransfer))]


    net = SimpleGAN(0, .1)
    net.train()
