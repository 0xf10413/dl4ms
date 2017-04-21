import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')
sys.path.append('../motion')

from Network import Network
from AdamTrainer import AdamTrainer
from network import create_core, create_regressor

rng = np.random.RandomState(23455)

data = np.load('../data/processed/data_edin_locomotion.npz')['clips']

I = np.arange(len(data))
rng.shuffle(I)

data_train = data[I[:len(data)//2]]
data_valid = data[I[len(data)//2:]]

X = data_train
X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')
X = (X - preprocess['Xmean']) / preprocess['Xstd']

Y = X[:,-7:]

I = np.arange(len(X))
rng.shuffle(I)
X, Y = X[I], Y[I]

print(X.shape, Y.shape)

batchsize = 1

network_second = create_core(batchsize=batchsize, window=X.shape[2], dropout=0.0, depooler=lambda x,**kw: x/2)
network_second.load(np.load('network_core.npz'))

network_first = create_regressor(batchsize=batchsize, window=X.shape[2], input=Y.shape[1])
network = Network(network_first, network_second[1], params=network_first.params)

X = theano.shared(X, borrow=True)
Y = theano.shared(Y, borrow=True)

trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=250, alpha=0.00001)
trainer.train(network, Y, X, filename='network_regression_flo.npz')
