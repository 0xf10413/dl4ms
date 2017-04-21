import os
import sys
import numpy as np
import theano
import theano.tensor as T
import theano.d3viz as d3v
from theano.printing import pydotprint

sys.path.append('../nn')
from network import *

rng = np.random.RandomState(23456)

""" Loading any type of data """
X = np.load('../data/processed/data_hdm05.npz')['clips']
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

E = theano.shared(X, borrow=True)

""" Creating the network (pick one) """
batchsize = 1
#network = create_core(rng=rng, batchsize=batchsize, window=X.shape[2])
#network = create_footstepper(batchsize=batchsize, window=X.shape[2], dropout=0.1)
network=create_regressor(batchsize=batchsize, window=X.shape[2], input=X.shape[1])

""" Drawing. Beware, html rendering is buggy """
""" Theano need a patch : is_to_equals.2017_04_21.patch """
to_print = network.getopgraph(E)
#d3v.d3viz(to_print, 'network.html')
pydotprint(to_print, 'network.png')
