import sys
import time
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from Network import Network
from network import create_core, create_regressor
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

rng = np.random.RandomState(23455)

data = np.load('../data/processed/data_hdm05.npz')
data_punching = np.hstack([np.arange(259, 303), np.arange(930, 978), np.arange(1650,1703), np.arange(2243,2290), np.arange(2851,2895)])
rng.shuffle(data_punching)

punching_train = data_punching[:len(data_punching)//2]
punching_valid = data_punching[len(data_punching)//2:]

X = data['clips'][punching_valid]
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

hands = np.array([60,61,62,48,49,50])

Y = X[:,hands]

batchsize = 1
window = X.shape[2]

network_first = create_regressor(batchsize=batchsize, window=window, input=Y.shape[1], dropout=0.0)
network_second = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw:x/2)
network_second.load(np.load('network_core.npz'))
network = Network(network_first, network_second[1], params=network_first.params)
network.load(np.load('network_regression_punch.npz'))

from AnimationPlot import animation_plot

for i in range(len(X)):

    network_func = theano.function([], network(Y[i:i+1]))

    Xorig = np.array(X[i:i+1])
    start = time.clock()
    Xrecn = network_func()
    Xorig = (Xorig * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = constrain(Xrecn, network_second[0], network_second[1], preprocess, multiconstraint(
        foot_sliding(Xrecn[:,-4:].copy()),
        joint_lengths()), alpha=0.01, iterations=50)
    
    animation_plot([Xorig, Xrecn], interval=15.15)

