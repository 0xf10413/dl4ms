#!/usr/bin/env python3
import os

import numpy as np
import scipy

import matplotlib.pyplot as plt
from matplotlib import animation

os.environ['THEANO_FLAGS'] = (
    #'device=cuda'
    'device=cpu'
)

import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
import lasagne.layers as ll
import lasagne.nonlinearities as lnl
import lasagne.init as lin

from simplegan import SimpleGAN

class UnboundGenerator(object):
    """
    Générateur simple, à deux couches FC+tanh
    N'est pas borné en sortie
    """
    def __init__(self, in_size, out_size):
        ## Construction du générateur
        self.layers = dict()
        self.layers['in'] = ll.InputLayer(
                shape=(None, in_size),
                )
        self.layers['hidden1'] = ll.DenseLayer(
                self.layers['in'], num_units=50,
                nonlinearity=lnl.tanh,
                W=lin.GlorotUniform(),
                b=lin.Constant(0),
                )
        self.layers['hidden2'] = ll.DenseLayer(
                self.layers['hidden1'], num_units=out_size,
                nonlinearity=None,
                W=lin.GlorotUniform(),
                b=lin.Constant(0),
                )
        self.layers['out'] = self.layers['hidden2']
        self.params = ll.get_all_params(self.layers['out'], trainable=True)

class UnboundDiscriminator(object):
    """
    Discrimineur simple, à trois couches FC+tanh
    Si l'input est de taille x, FC1 est de taille x*10 et FC2 x*20
    La sortie n'est pas bornée
    """
    def __init__(self, in_shape, out_shape):
        ## Construction du générateur
        ## in_shape : batch_size x flatten_size
        # Première version : évaluation sur X
        self.layers = dict()
        self.layers['in'] = ll.InputLayer(
                shape=(None, in_shape),
                )
        self.layers['hidden1'] = ll.DenseLayer(
                self.layers['in'], num_units=50,
                nonlinearity=lnl.elu,
                W=lin.GlorotUniform(),
                b=lin.Constant(0),
                )
        self.layers['hidden2'] = ll.DenseLayer(
                self.layers['hidden1'], num_units=50,
                nonlinearity=lnl.tanh,
                W=lin.GlorotUniform(),
                b=lin.Constant(0),
                )
        self.layers['hidden3'] = ll.DenseLayer(
                self.layers['hidden2'], num_units=out_shape,
                nonlinearity=None,
                W=lin.GlorotUniform(),
                b=lin.Constant(0),
                )
        self.layers['out'] = self.layers['hidden3']
        self.params = ll.get_all_params(self.layers['out'], trainable=True)


class WGAN(SimpleGAN):
    def __init__(self, rng, **kwargs):
        super().__init__(**kwargs)
        self.srng = RandomStreams(seed=rng.randint(0, np.iinfo(np.uint16).max))

    def prepare_training(self):
        """
        Compilation des fonctions d'entraînement
        """
        ## Fonction d'entraînement du générateur
        # Variables d'input
        z = T.tensor(dtype=config.floatX, broadcastable=(False, False))
        x = T.tensor(dtype=config.floatX, broadcastable=(False, False))
        x_fake = ll.get_output(self.G.layers['out'], inputs=z)

        # Objectifs
        score_true = ll.get_output(self.D.layers['out'], inputs=x)
        score_fake = ll.get_output(self.D.layers['out'], inputs=x_fake)

        batch_size = x.shape[0]
        t = self.srng.uniform((batch_size, )).dimshuffle(0, 'x')

        x_btwn = t * x + (1 - t) * x_fake
        score_btwn = ll.get_output(self.D.layers['out'], inputs=x_btwn)

        grad_btwn = theano.grad(score_btwn.sum(), wrt=x_btwn)
        grad_l2 = T.sqrt((grad_btwn ** 2).sum(axis=1))

        gamma = 0.1

        obj_g = -T.mean(score_fake)
        obj_d = -T.mean(score_true - score_fake - gamma * (grad_l2 - 1) ** 2)

        params_g = ll.get_all_params(self.G.layers['out'], trainable=True)
        params_d = ll.get_all_params(self.D.layers['out'], trainable=True)

        # variables symboliques, permet d'ajuster le learning rate pendant
        # l'apprentissage
        lr_g = T.scalar(dtype=config.floatX)
        lr_d = T.scalar(dtype=config.floatX)

        updates_g = lasagne.updates.rmsprop(obj_g, params_g, lr_g)
        updates_d = lasagne.updates.rmsprop(obj_d, params_d, lr_d)

        ## Compilation des fonctions d'entraînement
        print("Now compiling")
        self.train_g = theano.function(
            inputs=[z, lr_g],
            outputs=[obj_g],
            updates=updates_g,
            #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
        )

        self.train_d = theano.function(
            inputs=[z, x, lr_d],
            outputs=[obj_d],
            updates=updates_d,
            #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
        )

        self.generate_x = theano.function(
            inputs=[z],
            outputs=[ll.get_output(self.G.layers['out'], inputs=z, deterministic=True)],
            #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
        )

        self.score_x = theano.function(
            inputs=[x],
            outputs=[ll.get_output(self.D.layers['out'], inputs=x, deterministic=True)],
            #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
        )


if __name__ == "__main__":
    # Test rapide de l'entraînement
    import os
    os.environ['THEANO_FLAGS'] = (
        'device=cpu'
    )

    import numpy as np
    import lasagne

    from distributions import GaussianDistribution, UniformDistribution

    rng = np.random.RandomState(43)
    lasagne.random.set_rng(rng)

    z_size = 50
    x_size = 10

    G = UnboundGenerator(z_size, x_size)
    D = UnboundDiscriminator(x_size, 1)
    X = GaussianDistribution(rng, x_size, mean=0, scale=.2)
    Z = UniformDistribution(rng, z_size, a=-1, b=1)

    net = WGAN(rng, Generator=G, Discriminator=D, TrueDistribution=X, Inspiration=Z)
    net.train(rng)
