import os
import numpy as np
import theano
import theano.tensor as T
import theano.d3viz as d3v
from theano.printing import pydotprint

from Layer import Layer

class Network(Layer):

    def __init__(self, *layers, **kw):
        self.layers = layers

        if kw.get('params', None) is None:
            self.params = sum([layer.params for layer in self.layers], [])
        else:
            self.params = kw.get('params', None)

    def __call__(self, input):
        for layer in self.layers: input = layer(input)
        return input

    def __getitem__(self, k):
        return self.layers[k]

    def freeze(self):
        """Cache les paramètres, empêche l'apprentissage"""
        self.hidden_params = self.params
        self.params = []

    def thaw(self):
        """Rend l'apprentissage possible après un gel"""
        self.params = self.hidden_params

    def cost(self, input):
        costs = 0
        for layer in self.layers:
            costs += layer.cost(input)
            input = layer(input)
        return costs / len(self.layers) ## flo: pourquoi moyenner le coût ?

    def save(self, database, prefix=''):
        for li, layer in enumerate(self.layers):
            layer.save(database, '%sL%03i_' % (prefix, li))

    def load(self, database, prefix=''):
        for li, layer in enumerate(self.layers):
            layer.load(database, '%sL%03i_' % (prefix, li))

    def getopgraph(self, input):
        opgraph = None
        for layer in self.layers:
            if isinstance(layer, Network):
                input = layer.getopgraph(input)
            else:
              next_input = layer(input)
              opgraph = theano.OpFromGraph([input], [next_input],
                  name=type(layer).__name__)
              input = opgraph(input)
        return input

