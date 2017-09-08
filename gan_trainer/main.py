#!/usr/bin/env python3
import os
os.environ['THEANO_FLAGS'] = (
    #'device=cuda'
    'device=cpu'
)

import numpy as np
import lasagne

from simplegan import SimpleGAN, SimpleGenerator, SimpleDiscriminator
from wgan import WGAN, UnboundGenerator, UnboundDiscriminator
from distributions import (
        GaussianDistribution,
        UniformDistribution,
        LaplaceDistribution,
        GaussianMixture,
        )
#from dcgan import DCGAN

rng = np.random.RandomState(43)
lasagne.random.set_rng(rng)

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
    #Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
    #Xstyletransfer = np.swapaxes(Xstyletransfer, 1, 2).astype(theano.config.floatX)
    #preprocess = np.load('../synth/preprocess_core.npz')
    #Xstyletransfer = (Xstyletransfer - preprocess['Xmean']) / preprocess['Xstd']

    #pairings = [(i, Xstyletransfer) for i in range(len(Xstyletransfer))]

    z_size = 100
    x_size = 5
    gan_type = "simple"
    gan_type = "wgan"

    X = \
        + GaussianMixture(rng, x_size, means=[-.5, .5], scales=[.1, .1]) \
        #+ UniformDistribution(rng, x_size, a=-1, b=1)\
        #+ LaplaceDistribution(rng, x_size, mean=.1, scale=.1)\
        #+ GaussianDistribution(rng, x_size, mean=.2, scale=.1)\
        #+ GaussianDistribution(rng, x_size, mean=-.2, scale=.1)\
        #
    Z = GaussianDistribution(rng, z_size, mean=0, scale=.2)

    if gan_type == "simple":
        G = SimpleGenerator(z_size, x_size)
        D = SimpleDiscriminator(x_size, 1)
        net = SimpleGAN(Generator=G, Discriminator=D, TrueDistribution=X, Inspiration=Z)
    elif gan_type == "wgan":
        G = UnboundGenerator(z_size, x_size)
        D = UnboundDiscriminator(x_size, 1)
        net = WGAN(rng, Generator=G, Discriminator=D, TrueDistribution=X, Inspiration=Z)

    net.train(rng, max_epochs=500)

    from matplotlib import pyplot as plt
    from scipy.stats import skew, kurtosis
    plt.figure()
    distr = net.generate_x(Z.sample((100000, Z.sample_size)))
    plt.hist(distr, bins=500, normed=True, label="Estimated distr")
    plt.title("Mean={:.2}, var={:.2}, skew={:.2}, kurtosis={}".
            format(np.mean(distr), np.var(distr),
                skew(distr, axis=None), kurtosis(distr, axis=None))
            )
    x_axis = np.linspace(np.min(distr), np.max(distr), len(distr[0]))
    true_dist = X.pdf(x_axis)
    plt.plot(x_axis, true_dist, label="True distr")
    plt.legend()
    plt.show()
