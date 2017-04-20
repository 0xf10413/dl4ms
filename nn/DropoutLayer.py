import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from Layer import Layer

class DropoutLayer(Layer):

    def __init__(self, amount=0.3, rng=np.random):
        self.amount = amount
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.params = []

    def __call__(self, input):
        if self.amount > 0.0:
            return (input * self.theano_rng.binomial(
                size=input.shape, n=1, p=(1-self.amount),
                dtype=theano.config.floatX)) / (1-self.amount)
        else:
            return input
