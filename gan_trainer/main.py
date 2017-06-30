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

sys.path.append('../gram_trainer')
from train_all import (
        Cacher,
        generate_gram_samples,
        generate_normalized_gram,
        )

np.random.seed(43)

####################
## Modèle de GAN ##
####################
class GANGenerator(object):
    """
    """
    def __init__(self, in_shape, out_size):
        ## Construction du générateur
        ## in_shape : batch_size x source size
        self.Z = T.matrix('Z')
        self.layers = dict()
        self.layers['in'] = lasagne.layers.InputLayer(
                shape=in_shape,
                input_var=self.Z)
        self.layers['hidden1'] = lasagne.layers.DenseLayer(
                self.layers['in'], num_units=in_shape[1],
                nonlinearity=lasagne.nonlinearities.tanh,
                W=lasagne.init.GlorotUniform(),
                b=lasagne.init.Constant(0),
                )
        self.layers['hidden2'] = lasagne.layers.DenseLayer(
                self.layers['hidden1'], num_units=out_size,
                nonlinearity=lasagne.nonlinearities.tanh,
                W=lasagne.init.GlorotUniform(),
                b=lasagne.init.Constant(0),
                )
        self.layers['out'] = self.layers['hidden2']
        self.params = lasagne.layers.get_all_params(self.layers['out'])
        self.output = lasagne.layers.get_output(self.layers['out'])
        self.predict = theano.function(
                [self.Z],
                lasagne.layers.get_output(
                    self.layers['out'], deterministic=True)
                )

class GANDiscriminator(object):
    """
    """
    def __init__(self, in_shape, out_shape, G):
        ## Construction du générateur
        ## in_shape : batch_size x flatten_size
        ## G : générateur
        # Première version : évaluation sur X
        self.X = T.matrix('X')
        self.layers = dict()
        self.layers['in'] = lasagne.layers.InputLayer(
                shape=in_shape,
                input_var=self.X)
        self.layers['hidden1'] = lasagne.layers.DenseLayer(
                self.layers['in'], num_units=in_shape[1],
                nonlinearity=lasagne.nonlinearities.tanh,
                W=lasagne.init.GlorotUniform(),
                b=lasagne.init.Constant(0),
                )
        self.layers['hidden2'] = lasagne.layers.DenseLayer(
                self.layers['hidden1'], num_units=out_shape,
                nonlinearity=lasagne.nonlinearities.tanh,
                W=lasagne.init.GlorotUniform(),
                b=lasagne.init.Constant(0),
                )
        self.layers['out'] = self.layers['hidden2']
        self.params = lasagne.layers.get_all_params(self.layers['out'])
        self.output = lasagne.layers.get_output(self.layers['out'])
        self.predict = theano.function(
                [self.X],
                lasagne.layers.get_output(
                    self.layers['out'], deterministic=True)
                )
        # Rebelote avec D(G(Z))
        self.layers2 = dict()
        self.layers2['in'] = lasagne.layers.InputLayer(
                shape=in_shape,
                input_var=lasagne.layers.get_output(G.layers['out'])
                )
        self.layers2['hidden1'] = lasagne.layers.DenseLayer(
                self.layers2['in'], num_units=in_shape[1],
                nonlinearity=lasagne.nonlinearities.tanh,
                W=self.layers['hidden1'].W,
                b=self.layers['hidden1'].b,
                )
        self.layers2['hidden2'] = lasagne.layers.DenseLayer(
                self.layers2['hidden1'], num_units=out_shape,
                nonlinearity=lasagne.nonlinearities.tanh,
                W=self.layers['hidden2'].W,
                b=self.layers['hidden2'].b,
                )
        self.layers2['out'] = self.layers2['hidden2']
        self.params2 = lasagne.layers.get_all_params(self.layers2['out'])
        self.output2 = lasagne.layers.get_output(self.layers2['out'])
        self.predict2 = theano.function(
                [G.Z],
                lasagne.layers.get_output(
                    self.layers2['out'], deterministic=True)
                )

class SimpleGAN(object):
    def __init__(self):
        self.z_size = 20
        self.x_size = 1
        self.G = GANGenerator((None, self.z_size), self.x_size)
        print("Generator ready")
        self.D = GANDiscriminator((None, self.x_size), 1, self.G)
        print("Discriminator ready")

    def train(self):
        ## Fonction d'entraînement du générateur
        lr = 4e-2
        batchsize = 10
        max_epochs = 501
        d_steps = 4

        ## Objectifs
        prediction_d = self.D.output # Prediction sur X
        prediction_dg = self.D.output2 # Prediction sur D(Z)
        params_d = self.D.params
        params_g = self.D.params2
        #params_g = self.G.params
        obj_d = T.mean(T.log(prediction_d) + T.log(1 - prediction_dg))
        obj_g = T.mean(T.log(prediction_dg))

        ## updates
        updates_d = lasagne.updates.adam(-obj_d, params_d, lr)
        updates_g = lasagne.updates.adam(-obj_g, params_g, lr)
        train_g = theano.function([self.G.Z],
                obj_g,
                updates=updates_g,
                allow_input_downcast=True,
                #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                )
        train_d = theano.function([self.G.Z, self.D.X],
                obj_d,
                updates=updates_d,
                allow_input_downcast=True,
                #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
                )

        ## Output
        out_d = theano.function([self.D.X], self.D.output,
                allow_input_downcast=True)
        out_g = theano.function([self.G.Z], self.G.output,
                allow_input_downcast=True)
        out_dg = theano.function([self.G.Z], self.D.output2,
                allow_input_downcast=True)

        losses_d = np.zeros(max_epochs)
        losses_g = np.zeros(max_epochs)
        anims = list()

        state = None
        def draw_state(title, ii, samples=1000):
            nonlocal state
            opinion = np.zeros(samples)
            opinion2 = np.zeros(samples)
            generate = np.zeros(samples)
            space = np.linspace(-5, 5, samples)
            true_dist = scipy.stats.norm.pdf(space)
            for i in range(samples):
                opinion[i] = out_d(space[np.newaxis, i:i+1])
                generate[i] = np.mean(out_g(np.random.rand(batchsize, self.z_size)), axis=0)
                opinion2[i] = out_d(generate[np.newaxis, i:i+1])

            if state is None:
                state = dict()
                state['fig'] = plt.figure()
                state['ax1'] = plt.subplot(2,1,1)
                state['11'] = plt.plot(space, opinion, label='over an interval')
                state['12'] = plt.plot(generate, opinion2, 'x', label='over G')
                state['13'] = plt.plot(space, true_dist, label='true distribution')
                plt.ylim(-1,1)
                plt.legend()
                plt.title(title)

                state['ax'] = plt.subplot(2,1,2)
                state['21'] = plt.plot(losses_d, label="Loss D")
                state['22'] = plt.plot(losses_g, label="Loss G")
                state['2a'] = plt.gca()
                plt.title("Losses")
                plt.legend()
            else:
                state['11'][0].set_ydata(opinion)
                state['12'][0].set_xdata(generate)
                state['12'][0].set_ydata(opinion2)
                state['13'][0].set_ydata(true_dist)
                state['21'][0].set_ydata(losses_g)
                state['21'][0].set_xdata(range(max_epochs))
                state['22'][0].set_ydata(losses_d)
                state['22'][0].set_xdata(range(max_epochs))
                state['2a'].relim()
                state['2a'].autoscale_view()

            filename = 'matplotlib/' + str(ii).zfill(5) + '.png'
            plt.savefig(filename)

        ## Boucle d'apprentissage
        for i in range(max_epochs):
            losses_d_t = np.zeros(d_steps)
            for j in range(d_steps):
                x = np.random.normal(0, 1, (batchsize, self.x_size))
                z = 5*np.random.rand(batchsize, self.z_size)
                losses_d_t[j] = train_d(z, x)
            losses_d[i] = np.mean(losses_d_t)
            z = 5*np.random.rand(batchsize, self.z_size)
            losses_g[i] = train_g(z)
            draw_state("Epoch {}".format(i), i)
            if i % 100 == 0:
                print("End of epoch",i)





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


    net = SimpleGAN()
    net.train()
