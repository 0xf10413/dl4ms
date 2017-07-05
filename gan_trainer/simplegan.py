#!/usr/bin/env python3
"""
Premier modèle de GAN, basé directement sur la formule classique minimax
N'applique pas de technique particulière
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
import lasagne


class SimpleGenerator(object):
    """
    Générateur simple, à deux couches FC+tanh
    Si l'input est de taille z, FC1 est de taille z//2
    """
    def __init__(self, in_shape, out_size):
        ## Construction du générateur
        ## in_shape : batch_size x source size
        self.Z = T.matrix('Z')
        self.layers = dict()
        self.layers['in'] = lasagne.layers.InputLayer(
                shape=in_shape,
                input_var=self.Z)
        print("Generator : in ",lasagne.layers.get_output_shape(self.layers['in']))
        self.layers['hidden1'] = lasagne.layers.DenseLayer(
                self.layers['in'], num_units=in_shape[1]//2,
                nonlinearity=lasagne.nonlinearities.tanh,
                W=lasagne.init.GlorotUniform(),
                b=lasagne.init.Constant(0),
                )
        print("Generator : hidden1 ",lasagne.layers.get_output_shape(self.layers['hidden1']))
        self.layers['hidden2'] = lasagne.layers.DenseLayer(
                self.layers['hidden1'], num_units=out_size,
                nonlinearity=lasagne.nonlinearities.ScaledTanh(scale_out=5),
                W=lasagne.init.GlorotUniform(),
                b=lasagne.init.Constant(0),
                )
        print("Generator : hidden2 ",lasagne.layers.get_output_shape(self.layers['hidden2']))
        self.layers['out'] = self.layers['hidden2']
        self.params = lasagne.layers.get_all_params(self.layers['out'])
        self.output = lasagne.layers.get_output(self.layers['out'])
        self.predict = theano.function(
                [self.Z],
                lasagne.layers.get_output(
                    self.layers['out'], deterministic=True)
                )

class SimpleDiscriminator(object):
    """
    Discrimineur simple, à trois couches FC+tanh
    Si l'input est de taille x, FC1 est de taille x*10 et FC2 x*20
    """
    def __init__(self, in_shape, out_shape, G):
        ## Construction du générateur
        ## in_shape : batch_size x flatten_size
        ## G : générateur
        # Première version : évaluation sur X
        # ATTENTION : recopier la définition DEUX FOIS
        self.X = T.matrix('X')
        self.layers = dict()
        self.layers['in'] = lasagne.layers.InputLayer(
                shape=in_shape,
                input_var=self.X)
        print("Discriminator : in ",lasagne.layers.get_output_shape(self.layers['in']))
        self.layers['hidden1'] = lasagne.layers.DenseLayer(
                self.layers['in'], num_units=in_shape[1]*10,
                nonlinearity=lasagne.nonlinearities.tanh,
                W=lasagne.init.GlorotUniform(),
                b=lasagne.init.Constant(0),
                )
        print("Discriminator : hidden1",lasagne.layers.get_output_shape(self.layers['hidden1']))
        self.layers['hidden2'] = lasagne.layers.DenseLayer(
                self.layers['hidden1'], num_units=in_shape[1]*20,
                nonlinearity=lasagne.nonlinearities.tanh,
                W=lasagne.init.GlorotUniform(),
                b=lasagne.init.Constant(0),
                )
        print("Discriminator : hidden2",lasagne.layers.get_output_shape(self.layers['hidden2']))
        self.layers['hidden3'] = lasagne.layers.DenseLayer(
                self.layers['hidden2'], num_units=out_shape,
                nonlinearity=lasagne.nonlinearities.tanh,
                W=lasagne.init.GlorotUniform(),
                b=lasagne.init.Constant(0),
                )
        print("Discriminator : hidden3",lasagne.layers.get_output_shape(self.layers['hidden3']))
        self.layers['out'] = self.layers['hidden3']
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
        print("Discriminator2 : in ",lasagne.layers.get_output_shape(self.layers2['in']))
        self.layers2['hidden1'] = lasagne.layers.DenseLayer(
                self.layers2['in'], num_units=in_shape[1]*10,
                nonlinearity=lasagne.nonlinearities.tanh,
                W=self.layers['hidden1'].W,
                b=self.layers['hidden1'].b,
                )
        print("Discriminator2 : hidden1",lasagne.layers.get_output_shape(self.layers2['hidden1']))
        self.layers2['hidden2'] = lasagne.layers.DenseLayer(
                self.layers2['hidden1'], num_units=in_shape[1]*20,
                nonlinearity=lasagne.nonlinearities.tanh,
                W=self.layers['hidden2'].W,
                b=self.layers['hidden2'].b,
                )
        print("Discriminator2 : hidden2",lasagne.layers.get_output_shape(self.layers2['hidden2']))
        self.layers2['hidden3'] = lasagne.layers.DenseLayer(
                self.layers2['hidden2'], num_units=out_shape,
                nonlinearity=lasagne.nonlinearities.tanh,
                W=self.layers['hidden3'].W,
                b=self.layers['hidden3'].b,
                )
        print("Discriminator2 : hidden3",lasagne.layers.get_output_shape(self.layers2['hidden3']))
        self.layers2['out'] = self.layers2['hidden3']
        self.params2 = lasagne.layers.get_all_params(self.layers2['out'])
        self.output2 = lasagne.layers.get_output(self.layers2['out'])
        self.predict2 = theano.function(
                [G.Z],
                lasagne.layers.get_output(
                    self.layers2['out'], deterministic=True)
                )

class SimpleGAN(object):
    """
    Un simple GAN, qui tente d'apprendre une gaussienne
    """
    def __init__(self, true_mean, true_scale):
        self.z_size = 10
        self.x_size = 1
        self.true_mean = true_mean
        self.true_scale = true_scale
        self.G = SimpleGenerator((None, self.z_size), self.x_size)
        print("Generator ready")
        self.D = SimpleDiscriminator((None, self.x_size), 1, self.G)
        print("Discriminator ready")

    def train(self):
        ## Fonction d'entraînement du générateur
        lr_d = 5e-2
        lr_g = lr_d/150
        batchsize = 50
        max_epochs = 501
        d_steps = 4

        ## Objectifs
        prediction_d = self.D.output # Prediction sur X
        prediction_dg = self.D.output2 # Prediction sur D(Z)
        params_d = self.D.params
        #params_g = self.D.params2
        params_g = self.G.params
        tmp_d1 = (prediction_d+1)/2
        tmp_d2 = 1 - (prediction_dg+1)/2
        obj_d = T.mean(T.log(tmp_d1)) + T.mean(T.log(tmp_d2))
        tmp_g = (prediction_dg+1)/2
        obj_g = T.mean(T.log(tmp_g))

        ## updates
        updates_d = lasagne.updates.adam(-obj_d, params_d, lr_d)
        updates_g = lasagne.updates.adam(-obj_g, params_g, lr_g)
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

        state = None
        def draw_state(title, ii, samples=1000):
            nonlocal state
            opinion = np.zeros(samples)
            opinion2 = np.zeros(samples)
            generate = np.zeros(samples)
            space = np.linspace(-5, 5, samples)
            true_dist = scipy.stats.norm.pdf(space, loc=self.true_mean, scale=self.true_scale)
            true_dist /= np.max(true_dist)
            for i in range(samples):
                opinion[i] = (out_d(space[np.newaxis, i:i+1])+1)/2
                generate[i] = np.mean(out_g(np.random.rand(batchsize, self.z_size)), axis=0)
            hist = np.histogram(generate, density=True)
            hist = list(hist)
            hist[0] = [(hist[0][i], hist[0][i]) for i in range(len(hist[0]))]
            hist[0] = [item for it in hist[0] for item in it]
            hist[1] = [(hist[1][i], hist[1][i+1]) for i in range(len(hist[1])-1)]
            hist[1] = [item for it in hist[1] for item in it]
            hist[0] = np.array(hist[0])
            hist[1] = np.array(hist[1])
            hist[0] /= np.max(hist[0])

            if state is None:
                state = dict()
                state['fig'] = plt.figure()
                state['ax1'] = plt.subplot(2,1,1)
                state['11'] = plt.plot(space, opinion, label='D opinion')
                state['12'] = plt.plot(hist[1], hist[0], label='G estimation')
                state['13'] = plt.plot(space, true_dist, label='true distribution')
                state['14'] = plt.plot(space, np.repeat(.5, len(space)), '--')
                state['1a'] = plt.gca()
                plt.legend()
                plt.title("Distributions")

                state['ax'] = plt.subplot(2,1,2)
                state['21'] = plt.plot(losses_d, label="Objective D")
                state['22'] = plt.plot(losses_g, label="Objective G")
                state['2a'] = plt.gca()
                plt.title("Objectives")
                plt.xlabel("Epoch")
                plt.legend()
            else:
                state['11'][0].set_ydata(opinion)
                state['12'][0].set_xdata(hist[1])
                state['12'][0].set_ydata(hist[0])
                state['13'][0].set_ydata(true_dist)
                state['1a'].relim()
                state['1a'].autoscale_view()
                state['21'][0].set_ydata(losses_d)
                state['21'][0].set_xdata(range(max_epochs))
                state['22'][0].set_ydata(losses_g)
                state['22'][0].set_xdata(range(max_epochs))
                state['2a'].relim()
                state['2a'].autoscale_view()

            filename = 'matplotlib/' + str(ii).zfill(5) + '.png'
            plt.savefig(filename)

        ## Boucle d'apprentissage
        for i in range(max_epochs):
            losses_d_t = np.zeros(d_steps)
            for j in range(d_steps):
                x = np.random.normal(self.true_mean, self.true_scale, (batchsize, self.x_size))
                z = 5*np.random.rand(batchsize, self.z_size)
                #print("Losses Dg", out_dg(z))
                #print("Losses D", out_d(x))
                losses_d_t[j] = train_d(z, x)
            losses_d[i] = np.mean(losses_d_t)
            z = 5*np.random.rand(batchsize, self.z_size)
            #print("Losses G", out_g(z))
            losses_g[i] = train_g(z)
            draw_state("Epoch {}".format(i), i)
            if i % 100 == 0:
                print("End of epoch",i)

