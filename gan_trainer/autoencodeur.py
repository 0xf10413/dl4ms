#!/usr/bin/env python3
"""
Divers outils pour travailler avec les autoencodeurs
"""
import os
import pickle

import numpy as np
import scipy

import matplotlib.pyplot as plt
from matplotlib import animation

os.environ['THEANO_FLAGS'] = (
    'device=cuda'
    #'device=cpu'
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
import lasagne.updates as lu

from progressbar import ProgressBar

class AbstractLayer(object):
    def __init__(self):
        self.layers = {}

    def __str__(self):
        if self.layers.get('out') is None:
            raise NotImplementedError("Missing 'out' layer")
        myself = ""
        for l in ll.get_all_layers(self.layers['out']):
            myself += format(l.output_shape) + ' ' + format(l.__class__) + '\n'
        return myself

    def save(self, filename):
        if self.layers.get('out') is None:
            raise NotImplementedError("Missing 'out' layer")
        values = ll.get_all_param_values(self.layers['out'])
        with open(filename, 'wb') as f:
            pickle.dump(values, f)

    def load(self, filename):
        if self.layers.get('out') is None:
            raise NotImplementedError("Missing 'out' layer")
        with open(filename, 'rb') as f:
            values = pickle.load(f)
            ll.set_all_param_values(self.layers['out'], values)

class AEDirectLayer(AbstractLayer):
    """
    Une couche d'auto-encodeur, côté encodeur
    Note : in_size est un tuple (batch_size, features, temps)
    """
    def __init__(self, in_size, *, dropout, num_filters, filter_size):
        super().__init__()
        self.in_size = in_size
        self.dropout = dropout
        self.num_filters = num_filters
        self.filter_size = filter_size

        self.layers['in'] = ll.InputLayer(
                shape=in_size,
                )
        self.layers['dropout'] = ll.DropoutLayer(
                self.layers['in'],
                p=dropout,
                )
        self.layers['conv'] = ll.Conv1DLayer(
                self.layers['dropout'],
                num_filters=num_filters,
                filter_size=filter_size,
                pad='same',
                nonlinearity=lnl.rectify,
                )
        self.layers['pool'] = ll.MaxPool1DLayer(
                self.layers['conv'],
                pool_size=2,
                )
        self.layers['out'] = self.layers['pool']
        self.params = ll.get_all_params(self.layers['out'], trainable=True)

class RandomDepooler(ll.Layer):
    def __init__(self, rng, shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = rng
        self.out_shape = shape
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))
        self.depool_shape = (2,)

    def get_output_for(self, input, **kwargs):
        input = input.dimshuffle(0,1,2,'x').repeat(self.depool_shape[0], axis=3)
        output_mask = self.theano_rng.uniform(size=(
            input.shape[0],self.out_shape[1],
            self.out_shape[2]//self.depool_shape[0],
            self.depool_shape[0]),
            dtype=theano.config.floatX)
        output_mask = T.floor(output_mask / output_mask.max(axis=3).dimshuffle(0,1,2,'x'))
        tmp =  (output_mask * input).reshape(self.out_shape)
        return tmp

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], 2*input_shape[2])



class AEReverseLayer(AbstractLayer):
    """
    Une couche d'autoencodeur, côté décodeur
    Se construit à partir d'une couche d'encodeur
    """
    def __init__(self, rng, sibling):
        super().__init__()
        self.sibling = sibling
        in_size = sibling.in_size
        dp_shape = self.sibling.layers['out'].output_shape

        self.layers['in'] = ll.InputLayer(
                shape=sibling.layers['out'].output_shape,
                )
        self.layers['depool'] = RandomDepooler(
                rng,
                (dp_shape[0], dp_shape[1], 2*dp_shape[2]),
                self.layers['in'],
                )
        self.layers['dropout'] = ll.DropoutLayer(
                self.layers['depool'],
                p=sibling.dropout,
                )
        #self.layers['reshape1'] = ll.ReshapeLayer(
        #       self.layers['dropout'],
        #       ([0],[1],[2], -1),
        #       )
        self.layers['deconv'] = ll.Conv1DLayer(
                self.layers['dropout'],
                num_filters=in_size[1],
                filter_size=sibling.filter_size,
                #W=self.sibling.layers['conv'].W.dimshuffle(1,0,2)[:,:,::-1],
                pad='same',
                nonlinearity=None,
                )
        #self.layers['reshape2'] = ll.ReshapeLayer(
        #       self.layers['deconv'],
        #       ([0],[1],[2]),
        #       )
        self.layers['out'] = self.layers['deconv']
        self.params = ll.get_all_params(self.layers['out'], trainable=True)


class Cacher(object):
    def __init__(self, arg):
        """
        Utilise arg pour créer un dossier de cache unique
        """
        self.path = "cache/" + arg
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def retrieve(self, filename, format='pkl'):
        """
        Récupère le fichier numpy si possible, renvoie None sinon
        """
        filename = self.get_filename(filename)
        if not os.path.exists(filename):
            return None
        else:
            print("(Cacher: " + filename + " already done, skipping)")
        if format == 'npz':
            return np.load(filename, 'r')
        return filename

    def get_filename(self, filename, format='pkl'):
        """
        Transforme le nom de l'objet à stocker en un chemin complet
        """
        return self.path + "/" + filename + '.' + format

class Autoencoder(object):
    def __init__(self, in_shape, rng, *, cost):
        dropout = .25
        self.lr = .01
        self.rng = rng
        self.cc = Cacher("simple_run")
        self.encode = [] # encodage partiel
        self.decode = [] # décodage partiel
        self.redecode = [] # encodage puis décodage partiel
        self.trains = [] # fonctions d'entraînement partielles
        self.layers = dict()

        if cost == "mse":
            self.cost = lambda x, y: T.mean((x - y)**2)
        elif cost == "x-entropy":
            self.cost = lambda x, y: T.nnet.binary_crossentropy(x ,y).mean()

        self.layers['dl0'] = AEDirectLayer(
                in_shape,
                dropout=dropout,
                num_filters=256,
                filter_size=25,
                )
        self.layers['rl0'] = AEReverseLayer(self.rng, self.layers['dl0'])

        #self.layers['dl1'] = AEDirectLayer(
        #        (None, 64, in_shape[2]//2),
        #        dropout=dropout,
        #        num_filters=128,
        #        filter_size=15,
        #        )

        #self.layers['rl1'] = AEReverseLayer(self.rng, self.layers['dl1'])

        #self.layers['dl2'] = AEDirectLayer(
        #        (None, 128, in_shape[2]//4),
        #        dropout=dropout,
        #        num_filters=256,
        #        filter_size=15,
        #        )
        #self.layers['rl2'] = AEReverseLayer(self.rng, self.layers['dl2'])


    def __str__(self):
        return str(self.layers)


    def prepare_training(self, deterministic=False):
        if not deterministic:
            print("Preparing model for training")
        else:
            print("Preparing model for testing")
        self.encode = []
        self.decode = []
        self.redecode = []
        self.trains = []

        last_layer = len(self.layers)//2
        last_loop = False
        for i in range(last_layer+1):
            if i == last_layer:
                i -= 1
                last_loop = True

            X = T.tensor3(dtype=config.floatX)
            Y = T.tensor3(dtype=config.floatX)

            e = ['dl%d'%j for j in range(i+1)]
            d = ['rl%d'%j for j in range(i+1)]

            # Calcul de l'encodeur complet
            # Méthode : on part de l'input X, on descend
            layer_encoding = X
            for e_i in e:
                layer_encoding = ll.get_output(
                        self.layers[e_i].layers['out'],
                        inputs=layer_encoding,
                        deterministic=(deterministic or e_i != e[-1]),
                        )

            # Calcul du décodeur complet
            # Méthode : on part de l'input Y, on remonte
            layer_decoding = Y
            for d_i in reversed(d):
                layer_decoding = ll.get_output(
                        self.layers[d_i].layers['out'],
                        inputs=layer_decoding,
                        deterministic=(deterministic or d_i != d[0]),
                        )

            # Calcul de l'encodeur-décodeur complet
            layer_redecoding = X
            for e_i in e:
                layer_redecoding = ll.get_output(
                        self.layers[e_i].layers['out'],
                        inputs=layer_redecoding,
                        deterministic=(deterministic or e_i != e[-1]),
                        )
            for d_i in reversed(d):
                layer_redecoding = ll.get_output(
                        self.layers[d_i].layers['out'],
                        inputs=layer_redecoding,
                        deterministic=(deterministic or d_i != d[0]),
                        )

            self.encode.append(theano.function([X], layer_encoding))
            self.decode.append(theano.function([Y], layer_decoding))
            self.redecode.append(theano.function([X], layer_redecoding))

            rebuild_cost = self.cost(X, layer_redecoding)
            if not last_loop:
                params_e = self.layers[e[-1]].params
                params_d = self.layers[d[-1]].params
            else:
                params_e = sum([self.layers[e[j]].params for j in range(i+1)], [])
                params_d = sum([self.layers[d[j]].params for j in range(i+1)], [])
            params = params_e + params_d
            #rebuild_cost += .01*sum([T.mean(abs(p)) for p in params])
            rebuild_cost += .1*T.mean(abs(params_d[0]))
            updates = lu.adam(rebuild_cost, params, self.lr)
            self.trains.append(theano.function([X], rebuild_cost, updates=updates))


    def iterate_minibatches(self, inputs, batchsize, shuffle=False):
        if shuffle:
            indices = np.arange(len(inputs))
            self.rng.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt]

    def train_layer(self, i_layer, X_train, *, max_epochs, batch_size):
        for epoch in range(max_epochs):
            train_err = []
            pb = ProgressBar(max_value=len(X_train)//batch_size)
            for batch in pb(self.iterate_minibatches(
                    X_train,
                    batch_size,
                    shuffle=True)):
                train_err.append(self.trains[i_layer](batch))
            print("Epoch {}, train_err={}, validation_err=??".format(
                epoch, np.mean(train_err),
                ))

    def end_training(self):
        self.lr /= 100
        self.prepare_training(deterministic=True)


    def train(self, X_train, max_epochs=100, batch_size=100):
        last_layer = len(self.layers)//2
        print("=== Beggining training ===")
        for i in range(last_layer):
            print("== Training layer {} ==".format(i))
            e = 'dl%d'%i
            d = 'rl%d'%i
            encoder = self.cc.retrieve(e)
            decoder = self.cc.retrieve(d)
            if encoder is None or decoder is None:
                self.train_layer(i, X_train, max_epochs=max_epochs, batch_size=batch_size)
                self.layers[e].save(self.cc.get_filename(e))
                self.layers[d].save(self.cc.get_filename(d))
            else:
                self.layers[e].load(encoder)
                self.layers[d].load(decoder)

        if len(self.layers) <= 2:
            print("== No fine tuning, not enough layers ==")
            return
        print("== Fine-tuning everyone ==")
        # Le fine-tuning a-t-il déjà été fait ?
        d_retrieve = [self.cc.retrieve('dl%d_ft'%i) for i in range(last_layer)]
        e_retrieve = [self.cc.retrieve('rl%d_ft'%i) for i in range(last_layer)]
        if any(d_retrieve) or any(e_retrieve):
            # Réponse : peut-être
            if all(d_retrieve) and all(e_retrieve):
                # Réponse: oui
                for i, d_r in enumerate(d_retrieve):
                    self.layers['dl%d'%i].load(d_r)
                for i, e_r in enumerate(e_retrieve):
                    self.layers['rl%d'%i].load(e_r)
            else:
                raise RuntimeError("Error ! Cache might be corrupted !")
        else:
            # Réponse : non
            self.train_layer(last_layer, X_train,
                    max_epochs=max_epochs, batch_size=batch_size)
            for i in range(last_layer):
                e = 'dl%d'%i
                d = 'rl%d'%i
                self.layers[e].save(self.cc.get_filename(e+'_ft'))
                self.layers[d].save(self.cc.get_filename(d+'_ft'))
        print("== End of fine-tuning ==")
        print("=== End of training ===")
        self.end_training()



if __name__ == "__main__":
    #from mnist import load_dataset as load_dataset_mnist
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_mnist()
    #x_shape = (None, X_train[2], X_train[3])

    from sys import path
    path.append("../nn")
    from AnimationPlot import animation_plot

    Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
    Xstyletransfer = np.swapaxes(Xstyletransfer, 1, 2).astype(theano.config.floatX)
    Xedin_punching = np.load('../data/processed/data_edin_punching.npz')['clips']
    Xedin_punching = np.swapaxes(Xedin_punching, 1, 2).astype(theano.config.floatX)
    Xedin_locomotion = np.load('../data/processed/data_edin_locomotion.npz')['clips']
    Xedin_locomotion = np.swapaxes(Xedin_locomotion, 1, 2).astype(theano.config.floatX)
    Xcmu = np.load('../data/processed/data_cmu.npz')['clips']
    Xcmu = np.swapaxes(Xcmu, 1, 2).astype(theano.config.floatX)

    preprocess = np.load('../synth/preprocess_core.npz')
    Xstyletransfer = (Xstyletransfer - preprocess['Xmean']) / preprocess['Xstd']
    Xedin_punching = (Xedin_punching - preprocess['Xmean']) / preprocess['Xstd']
    Xedin_locomotion = (Xedin_locomotion - preprocess['Xmean']) / preprocess['Xstd']
    Xcmu = (Xcmu - preprocess['Xmean']) / preprocess['Xstd']
    X_train = np.concatenate([Xedin_locomotion, Xcmu[:50]], axis=0)
    #X_train = np.concatenate([Xedin_locomotion], axis=0)
    #X_train = np.concatenate([Xedin_locomotion[:10]], axis=0)

    batch_size = 1
    x_shape = (batch_size, 73, 240)


    rng = np.random.RandomState(23456)

    A = Autoencoder(x_shape, rng, cost="mse")
    A.prepare_training()
    for i in ["dl0", "dl1", "dl2", "rl2", "rl1" , "rl0"]:
        print(i, A.layers.get(i))

    #A.train(X_train[:1,0,...], max_epochs=5, batch_size=500)
    #A.train(X_train, max_epochs=150, batch_size=100)
    A.train(X_train, max_epochs=100, batch_size=1)

    x = X_train[:1]
    x = A.redecode[-1](X_train[:1])
    x = (x*preprocess['Xstd']) + preprocess['Xmean']
    y = (X_train[:1]*preprocess['Xstd']) + preprocess['Xmean']
    print(x.shape)
    animation_plot([y, x], interval=15.15)

    #for i in A.iterate_minibatches(X_val[:,0,...], 10):
    #    plt.figure()
    #    for j in range(10):
    #        plt.subplot(10,2,2*j+1)
    #        plt.imshow(i[j], cmap=plt.get_cmap("gray"))
    #        plt.gca().set_aspect('auto')
    #        plt.subplot(10,2,2*j+2)
    #        plt.imshow(A.redecode[-1](i[j:j+1])[0], cmap=plt.get_cmap("gray"))
    #        plt.gca().set_aspect('auto')
    #    plt.subplots_adjust(wspace=0, hspace=0)
    #    plt.show()

