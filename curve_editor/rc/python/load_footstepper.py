#!/usr/bin/env python3
"""
Chargement du footstepper
Ajoute aux données initiales les moments où les pieds touchent le sol
"""

# Imports supplémentaires
from Network import Network
from network import create_core, create_regressor, create_footstepper
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

# Chargement des données
rng = np.random.RandomState(23455)
preprocess = np.load('../synth/preprocess_core.npz')
preprocess_footstepper = np.load('../synth/preprocess_footstepper.npz')
batchsize = 1

# Fonction de création du réseau
def create_network(window, input):
    network_first = create_regressor(batchsize=batchsize, window=window, input=input, dropout=0.0)
    network_second = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw:x/2)
    network_second.load(np.load('../synth/network_core.npz'))
    network = Network(network_first, network_second[1], params=network_first.params)
    network.load(np.load('../synth/network_regression.npz'))
    return network_first, network_second, network

# Création du footstepper
input = theano.tensor.ftensor3()
Torig = curve
Torig = np.expand_dims(Torig, axis=0)
Torig = (Torig - preprocess['Xmean'][:,-7:-4]) / preprocess['Xstd'][:,-7:-4]

network_footstepper = create_footstepper(batchsize=batchsize, window=Torig.shape[0], dropout=0.0)
network_footstepper.load(np.load('../synth/network_footstepper.npz'))
network_footstepper_func = theano.function([input], network_footstepper(input), allow_input_downcast=True)

W = network_footstepper_func(Torig[:,:3])
W = (W * preprocess_footstepper['Wstd']) + preprocess_footstepper['Wmean']
W = W.astype(np.float32)

alpha, beta = 1.0, 0.0
minstep, maxstep = 0.9, -0.5
off_lh, off_lt, off_rh, off_rt = 0.0, -0.1, np.pi+0.0, np.pi-0.1
Torig = (np.concatenate([Torig,
    (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_lh)>np.clip(np.cos(W[:,1:2])+beta, maxstep, minstep)).astype(theano.config.floatX)*2-1,
    (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_lt)>np.clip(np.cos(W[:,2:3])+beta, maxstep, minstep)).astype(theano.config.floatX)*2-1,
    (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_rh)>np.clip(np.cos(W[:,3:4])+beta, maxstep, minstep)).astype(theano.config.floatX)*2-1,
    (np.sin(np.cumsum(alpha*W[:,0:1],axis=2)+off_rt)>np.clip(np.cos(W[:,4:5])+beta, maxstep, minstep)).astype(theano.config.floatX)*2-1], axis=1))
