import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from network import create_core
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

rng = np.random.RandomState(23455)

preprocess = np.load('preprocess_core.npz')

batchsize = 1
window = 240

network = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
network.load(np.load('network_core.npz'))

from AnimationPlot import animation_plot

for i in range(10):

    Xbasis0 = np.zeros((1, 256, window//2), dtype=theano.config.floatX)
    Xbasis1 = np.zeros((1, 256, window//2), dtype=theano.config.floatX)
    Xbasis2 = np.zeros((1, 256, window//2), dtype=theano.config.floatX)

    Xbasis0[:,i*3+0] = 1 + 2 * np.sin(np.linspace(0.0, np.pi*8, window//2))
    Xbasis1[:,i*3+1] = 1 + 2 * np.sin(np.linspace(0.0, np.pi*8, window//2))
    Xbasis2[:,i*3+2] = 1 + 2 * np.sin(np.linspace(0.0, np.pi*8, window//2))

    Xbasis0 = np.array(network[1](theano.shared(Xbasis0, borrow=True)).eval())
    Xbasis1 = np.array(network[1](theano.shared(Xbasis1, borrow=True)).eval())
    Xbasis2 = np.array(network[1](theano.shared(Xbasis2, borrow=True)).eval())

    Xbasis0 = (Xbasis0 * preprocess['Xstd']) + preprocess['Xmean']
    Xbasis1 = (Xbasis1 * preprocess['Xstd']) + preprocess['Xmean']
    Xbasis2 = (Xbasis2 * preprocess['Xstd']) + preprocess['Xmean']

    animation_plot([Xbasis0, Xbasis1, Xbasis2], interval=15.15)

