#!/usr/bin/env python3
"""
Premier modèle de GAN, basé directement sur la formule classique minimax
N'applique pas de technique particulière
Corrigé par Maxime Gasse
"""
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

import lasagne
import lasagne.layers as ll
import lasagne.nonlinearities as lnl
import lasagne.init as lin


class SimpleGenerator(object):
    """
    Générateur simple, à deux couches FC+tanh
    """
    def __init__(self, in_size, out_size):
        ## Construction du générateur
        self.layers = dict()
        self.layers['in'] = ll.InputLayer(
                shape=(None, in_size),
                )
        self.layers['hidden1'] = ll.DenseLayer(
                self.layers['in'], num_units=10,
                nonlinearity=lnl.tanh,
                W=lin.GlorotUniform(),
                b=lin.Constant(0),
                )
        self.layers['hidden2'] = ll.DenseLayer(
                self.layers['hidden1'], num_units=out_size,
                nonlinearity=lnl.tanh,
                W=lin.GlorotUniform(),
                b=lin.Constant(0),
                )
        self.layers['out'] = self.layers['hidden2']
        self.params = ll.get_all_params(self.layers['out'], trainable=True)

class SimpleDiscriminator(object):
    """
    Discrimineur simple, à trois couches FC+tanh
    Si l'input est de taille x, FC1 est de taille x*10 et FC2 x*20
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
                self.layers['in'], num_units=10,
                nonlinearity=lnl.tanh,
                W=lin.GlorotUniform(),
                b=lin.Constant(0),
                )
        self.layers['hidden2'] = ll.DenseLayer(
                self.layers['hidden1'], num_units=10,
                nonlinearity=lnl.tanh,
                W=lin.GlorotUniform(),
                b=lin.Constant(0),
                )
        self.layers['hidden3'] = ll.DenseLayer(
                self.layers['hidden2'], num_units=out_shape,
                nonlinearity=lnl.sigmoid,
                W=lin.GlorotUniform(),
                b=lin.Constant(0),
                )
        self.layers['out'] = self.layers['hidden3']
        self.params = ll.get_all_params(self.layers['out'], trainable=True)

class Distribution(object):
    def __init__(self, rng, sample_size):
        self.rng = rng
        self.sample_size = sample_size
        self._pdf = None

    def sample(self, sample_size):
        raise NotImplementedError("Did not extend base method")

    def pdf(self, interval):
        if self._pdf is None:
            raise NotImplementedError("No known pdf for this distribution")
        return self._pdf(interval)

class GaussianDistribution(Distribution):
    def __init__(self, rng, sample_size, *, mean, scale):
        super().__init__(rng, sample_size)
        self.mean = mean
        self.scale = scale
        self._pdf = lambda x: scipy.stats.norm.pdf(x, loc=self.mean,
                scale=self.scale).astype(config.floatX)

    def sample(self, sample_size):
        return rng.normal(loc=self.mean, scale=self.scale,
                size=sample_size).astype(config.floatX)

class UniformDistribution(Distribution):
    def __init__(self, rng, sample_size, *, a, b):
        super().__init__(rng, sample_size)
        assert a != b
        self.min = min(a, b)
        self.max = max(a, b)
        self._pdf = lambda x: 1/(self.max-self.min) if self.min <= x <= self.max else 0
        self._pdf = np.vectorize(self._pdf)

    def sample(self, sample_size):
        return rng.uniform(low=self.min, high=self.max,
                size=sample_size).astype(config.floatX)


class SimpleGAN(object):
    """
    Un simple GAN, sans amélioration spécifique
    """
    def __init__(self, *, Generator, Discriminator, TrueDistribution, Inspiration):
        self.G = Generator
        self.D = Discriminator
        self.X = TrueDistribution
        self.Z = Inspiration

        self.train_d = None
        self.train_g = None
        self.generate_x = None
        self.score_x = None

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

        obj_g = -T.mean(T.log(score_fake))
        obj_d = -T.mean(T.log(score_true) + T.log(1 - score_fake))

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

    def train(self, rng, *, batchsize=50, max_epochs=501, d_steps=10, g_steps=1):
        if not all([self.train_d, self.train_g, self.generate_x, self.score_x]):
            self.prepare_training()
        print("Now training")
        losses_d = np.zeros(max_epochs)
        losses_g = np.zeros(max_epochs)

        # Figure setup
        fig, sub = plt.subplots(nrows=2, ncols=1, squeeze=False)
        x_range = (-5, 5)
        y_range = (0, 1)
        n_samples = 1000
        x_axis = np.linspace(x_range[0], x_range[1], n_samples)

        true_dist = None
        true_dist = self.X.pdf(x_axis)
        dist_norm = np.max(true_dist)
        true_dist = true_dist / dist_norm  # normalization

        fake_dist, fake_bins = np.histogram(0, bins=n_samples - 1, range=x_range)
        fake_dist = np.repeat(fake_dist, 2)
        fake_bins = np.repeat(fake_bins, 2)[1:-1]

        curves = {}
        sub[0, 0].set_title("Distributions")
        sub[0, 0].set_xlim(x_range)
        sub[0, 0].set_ylim(y_range)
        sub[0, 0].plot(x_range, (0.5, 0.5), '--')
        sub[0, 0].plot(
            x_axis, true_dist, label='true distribution')
        curves['g_dist'] = sub[0, 0].plot(
            fake_bins, fake_dist, label='G estimation')[0]
        curves['d_score'] = sub[0, 0].plot(
            x_axis, np.repeat(0, n_samples), label='D opinion')[0]
        sub[0, 0].legend()

        sub[1, 0].set_title("Objectives")
        sub[1, 0].set_xlabel("Epoch")
        curves['d_obj'] = sub[1, 0].plot(
            range(max_epochs), np.repeat(0, max_epochs), label="Objective D")[0]
        curves['g_obj'] = sub[1, 0].plot(
            range(max_epochs), np.repeat(0, max_epochs), label="Objective G")[0]
        sub[1, 0].legend()

        plt.ion()
        plt.show()
        plt.draw()
        plt.pause(.01)

        writer = animation.FFMpegWriter()
        with writer.saving(fig, 'plot.mp4', dpi=300):

            # Boucle d'apprentissage
            d_objs = np.zeros(max_epochs)
            g_objs = np.zeros(max_epochs)
            for i in range(max_epochs):
                lr_d = 1e-3
                lr_g = lr_d

                print("epoch {} / {}...".format(i + 1, max_epochs))

                # train discriminator
                x = self.X.sample((d_steps, batchsize, self.X.sample_size))
                z = self.Z.sample((d_steps, batchsize, self.Z.sample_size))
                d_objs[i] = np.mean([self.train_d(z[j], x[j], lr_d)
                                     for j in range(d_steps)])

                # train generator
                z = self.Z.sample((g_steps, batchsize, self.Z.sample_size))
                g_objs[i] = np.mean([self.train_g(z[j], lr_g)
                                     for j in range(g_steps)])

                # plot indicators
                sub[0, 0].set_title('epoch {} / {}'.format(i + 1, max_epochs))

                curves['d_obj'].set_ydata(d_objs)
                curves['g_obj'].set_ydata(g_objs)

                sub[1, 0].relim()
                sub[1, 0].autoscale_view()

                x = x_axis[:, None].astype(config.floatX)
                scores = self.score_x(x)[0]
                curves['d_score'].set_ydata(scores)

                z = self.Z.sample((10*n_samples, self.Z.sample_size))
                fake_samples = self.generate_x(z)[0]
                fake_dist = np.histogram(fake_samples,
                                         bins=n_samples - 1, range=x_range,
                                         density=True)[0]
                fake_dist = np.repeat(fake_dist, 2)
                fake_dist = fake_dist / dist_norm
                curves['g_dist'].set_ydata(fake_dist)

                # write to video
                writer.grab_frame()

                # live display
                plt.draw()
                plt.pause(.01)

        plt.ioff()
        plt.show()

if __name__ == "__main__":
    # Test rapide de l'entraînement
    import os
    os.environ['THEANO_FLAGS'] = (
        'device=cpu'
    )

    import numpy as np
    import lasagne

    rng = np.random.RandomState(43)
    lasagne.random.set_rng(rng)

    z_size = 50
    x_size = 1

    G = SimpleGenerator(z_size, x_size)
    D = SimpleDiscriminator(x_size, 1)
    X = GaussianDistribution(rng, x_size, mean=0, scale=.2)
    Z = UniformDistribution(rng, z_size, a=-1, b=1)

    net = SimpleGAN(Generator=G, Discriminator=D, TrueDistribution=X, Inspiration=Z)
    net.train(rng)
