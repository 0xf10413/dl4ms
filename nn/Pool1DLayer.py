import os
import sys
import random
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from Layer import Layer

class Pool1DLayer(Layer):

    def __init__(self, input_shape, pool_shape=(2,), pooler=T.max):
        self.input_shape = input_shape
        self.pool_shape = pool_shape
        self.pooler = pooler
        self.params = []

    def __call__(self, input):
        return self.pooler(input.reshape((
            self.input_shape[0],self.input_shape[1],
            self.input_shape[2]//self.pool_shape[0],self.pool_shape[0])), axis=3)

