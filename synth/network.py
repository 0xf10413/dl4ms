#!/usr/bin/env python3
import sys
import numpy as np
import theano
import theano.tensor as T

sys.path.append('../nn')

from BiasLayer import BiasLayer
from Conv1DLayer import Conv1DLayer
from ActivationLayer import ActivationLayer
from DropoutLayer import DropoutLayer
from Pool1DLayer import Pool1DLayer
from Depool1DLayer import Depool1DLayer
from Network import Network

""" Build Network """

def create_core(rng=np.random, batchsize=1, window=240, dropout=0.25, depooler='random'):
    
    return Network(
        Network(
            DropoutLayer(amount=dropout, rng=rng),
            Conv1DLayer(filter_shape=(256, 73, 25), input_shape=(batchsize, 73, window), rng=rng),
            BiasLayer(shape=(256, 1)),
            ActivationLayer(),
            Pool1DLayer(input_shape=(batchsize, 256, window)),
        ),
        
        Network(
            Depool1DLayer(output_shape=(batchsize, 256, window), depooler='random', rng=rng),
            DropoutLayer(amount=dropout, rng=rng),
            Conv1DLayer(filter_shape=(73, 256, 25), input_shape=(batchsize, 256, window), rng=rng),
            BiasLayer(shape=(73, 1))
        )
    )

    
def create_regressor(rng=np.random, batchsize=1, window=240, input=4, dropout=0.25):
    
    return Network(
        DropoutLayer(amount=dropout, rng=rng),
        Conv1DLayer(filter_shape=(64, input, 45), input_shape=(batchsize, input, window), rng=rng),
        BiasLayer(shape=(64, 1)),
        ActivationLayer(),
    
        DropoutLayer(amount=dropout, rng=rng),
        Conv1DLayer(filter_shape=(128, 64, 25), input_shape=(batchsize, 64, window), rng=rng),
        BiasLayer(shape=(128, 1)),
        ActivationLayer(),

        DropoutLayer(amount=dropout, rng=rng),
        Conv1DLayer(filter_shape=(256, 128, 15), input_shape=(batchsize, 128, window), rng=rng),
        BiasLayer(shape=(256, 1)),
        ActivationLayer(),
        Pool1DLayer(input_shape=(batchsize, 256, window))
    )
        
def create_footstepper(rng=np.random, batchsize=1, window=250, dropout=0.25):
    
    return Network(
        DropoutLayer(amount=dropout, rng=rng),
        Conv1DLayer(filter_shape=(64, 3, 65), input_shape=(batchsize, 3, window), rng=rng),
        BiasLayer(shape=(64, 1)),
        ActivationLayer(),

        DropoutLayer(amount=dropout, rng=rng),
        Conv1DLayer(filter_shape=(5, 64, 45), input_shape=(batchsize, 64, window), rng=rng),
        BiasLayer(shape=(5, 1)),
    )
        
