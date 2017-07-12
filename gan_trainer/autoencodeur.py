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


class AEReverseLayer(AbstractLayer):
    """
    Une couche d'autoencodeur, côté décodeur
    Se construit à partir d'une couche d'encodeur
    """
    def __init__(self, sibling):
        super().__init__()
        self.sibling = sibling
        in_size = sibling.in_size

        self.layers['in'] = ll.InputLayer(
                shape=sibling.layers['out'].output_shape,
                )
        self.layers['depool'] = ll.Upscale1DLayer(
                self.layers['in'],
                scale_factor=2,
                )
        self.layers['dropout'] = ll.DropoutLayer(
                self.layers['depool'],
                p=sibling.dropout,
                )
        self.layers['reshape1'] = ll.ReshapeLayer(
               self.layers['dropout'],
               ([0],[1],[2], -1),
               )
        self.layers['deconv'] = ll.TransposedConv2DLayer(
                self.layers['reshape1'],
                num_filters=in_size[1],
                filter_size=sibling.filter_size,
                crop='same',
                nonlinearity=None,
                )
        self.layers['reshape2'] = ll.ReshapeLayer(
               self.layers['deconv'],
               ([0],[1],[2]),
               )
        self.layers['out'] = self.layers['reshape2']
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
        dropout = .5
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
                (None, X_train.shape[2], X_train.shape[3]),
                dropout=dropout,
                num_filters=64,
                filter_size=15
                )
        self.layers['rl0'] = AEReverseLayer(self.layers['dl0'])

        self.layers['dl1'] = AEDirectLayer(
                (None, 64, X_train.shape[3]//2),
                dropout=dropout,
                num_filters=128,
                filter_size=15
                )
        self.layers['rl1'] = AEReverseLayer(self.layers['dl1'])

    def __str__(self):
        return str(self.layers)


    def prepare_training(self):
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
                        inputs=layer_encoding
                        )

            # Calcul du décodeur complet
            # Méthode : on part de l'input Y, on remonte
            layer_decoding = Y
            for d_i in reversed(d):
                layer_decoding = ll.get_output(
                        self.layers[d_i].layers['out'],
                        inputs=layer_decoding
                        )

            # Calcul de l'encodeur-décodeur complet
            layer_redecoding = X
            for e_i in e:
                layer_redecoding = ll.get_output(
                        self.layers[e_i].layers['out'],
                        inputs=layer_redecoding
                        )
            for d_i in reversed(d):
                layer_redecoding = ll.get_output(
                        self.layers[d_i].layers['out'],
                        inputs=layer_redecoding
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
            updates = lu.adam(rebuild_cost, params_e + params_d, .01)
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

    def train_layer(self, i_layer, X_train, *, max_epochs=50, batch_size=100):
        for epoch in range(max_epochs):
            train_err = 0
            pb = ProgressBar(max_value=len(X_train)//batch_size)
            for batch in pb(self.iterate_minibatches(
                    X_train,
                    batch_size,
                    shuffle=True)):
                train_err += self.trains[i_layer](batch)
            print("Epoch {}, train_err={}, validation_err=??".format(
                epoch, train_err,
                ))

    def train(self, X_train, max_epochs=50, batch_size=100):
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



if __name__ == "__main__":
    from mnist import load_dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    x_shape = (None, X_train[2], X_train[3])

    rng = np.random.RandomState(43)

    A = Autoencoder(x_shape, rng, cost="mse")
    A.prepare_training()
    #for i in ["dl0", "dl1", "rl1" , "rl0"]:
    #    print(i, A.layers[i])

    A.train(X_train[:1,0,...])

    for i in A.iterate_minibatches(X_val[:,0,...], 10):
        plt.figure()
        plt.imshow(i[0])
        plt.figure()
        plt.imshow(A.redecode[-1](i[0:1])[0])
        plt.show()

