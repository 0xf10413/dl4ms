import os
import numpy as np
import theano
import theano.tensor as T

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
    
    def cost(self, input):
        costs = 0
        for layer in self.layers:
            costs += layer.cost(input)
            input = layer(input)
        return costs / len(self.layers)
    
    def save(self, database, prefix=''):
        for li, layer in enumerate(self.layers):
            layer.save(database, '%sL%03i_' % (prefix, li))
        
    def load(self, database, prefix=''):
        for li, layer in enumerate(self.layers):
            layer.load(database, '%sL%03i_' % (prefix, li))
        