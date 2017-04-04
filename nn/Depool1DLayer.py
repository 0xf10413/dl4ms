import os
import sys
import random
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from Layer import Layer

class Depool1DLayer(Layer):
    
    def __init__(self, output_shape, depool_shape=(2,), depooler='random', rng=np.random):
        
        self.depool_shape = depool_shape
        self.output_shape = output_shape
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.depooler = depooler
        self.params = []
        
    def __call__(self, input):
        
        input = input.dimshuffle(0,1,2,'x').repeat(self.depool_shape[0], axis=3)
        
        if self.depooler == 'random':
        
            output_mask = self.theano_rng.uniform(size=(
                self.output_shape[0],self.output_shape[1],
                self.output_shape[2]//self.depool_shape[0], self.depool_shape[0]),
                dtype=theano.config.floatX) 
            output_mask = T.floor(output_mask / output_mask.max(axis=3).dimshuffle(0,1,2,'x'))
            return (output_mask * input).reshape(self.output_shape)
            
        elif self.depooler == 'first':
        
            output_mask_np = np.zeros(self.depool_shape, dtype=theano.config.floatX)
            output_mask_np[0] = 1.0
            output_mask = theano.shared(mask_np, borrow=True).dimshuffle('x','x','x',0)
            return (output_mask * input).reshape(self.output_shape)
            
        else:
        
            return self.depooler(input, axis=3).reshape(self.output_shape)


        