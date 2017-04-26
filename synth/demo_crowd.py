import sys
import time
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')
sys.path.append('../motion')

from Network import Network
from theano.printing import pydotprint
from network import create_core, create_regressor, create_footstepper
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

from Quaternions import Quaternions

rng = np.random.RandomState(23455)

data = np.load('../data/crowddata.npz')

preprocess = np.load('preprocess_core.npz')
preprocess_footstepper = np.load('preprocess_footstepper.npz')

def create_network(batchsize, window, hidden):
    network_first = create_regressor(batchsize=batchsize, window=window, input=hidden, dropout=0.0, rng=rng)
    network_second = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw:x/2, rng=rng)
    network_second.load(np.load('network_core.npz'))
    network = Network(network_first, network_second[1], params=network_first.params)
    network.load(np.load('network_regression.npz'))
    return network_first, network_second, network

from AnimationPlot import animation_plot

scenes = [
    ('scene01', 500, 1100),
    ('scene02', 500, 1100),
    ('scene03', 500, 1100),
#    ('scene04', 500, 1100)
]

for scene, cstart, cend in scenes:

    input = theano.tensor.ftensor3()

    T = np.swapaxes(data[scene+'_Y'], 1, 2)
    T = T[:,:,cstart:cend].astype(theano.config.floatX)
    T = (T - preprocess['Xmean'][:,-7:-4]) / preprocess['Xstd'][:,-7:-4]
    T = np.concatenate([T, np.zeros((T.shape[0], 4, T.shape[2]), dtype=theano.config.floatX)], axis=1)

    #############

    network_footstepper = create_footstepper(batchsize=T.shape[0], window=T.shape[2], dropout=0.0, rng=rng)
    network_footstepper.load(np.load('network_footstepper.npz'))
    network_footstepper_func = theano.function([input], network_footstepper(input), allow_input_downcast=True)

    start = time.clock()
    W = network_footstepper_func(T[:,:3])
    W = (W * preprocess_footstepper['Wstd']) + preprocess_footstepper['Wmean']

    offsetvar = 2*np.pi*rng.uniform(size=(T.shape[0],1,1))
    stepvar = 1.0+0.05*rng.uniform(low=-1,high=1,size=(T.shape[0],1,1))
    thesvar = 0.05*rng.uniform(low=-1,high=1,size=(T.shape[0],1,1))
    off_lh, off_lt, off_rh, off_rt = 0.0, -0.25, np.pi+0.0, np.pi-0.25
    T[:,3:4] = (np.sin(np.cumsum(W[:,0:1]*stepvar,axis=2)+off_lh+offsetvar)>thesvar+np.cos(W[:,1:2])).astype(np.float)*2-1
    T[:,4:5] = (np.sin(np.cumsum(W[:,0:1]*stepvar,axis=2)+off_lt+offsetvar)>thesvar+np.cos(W[:,2:3])).astype(np.float)*2-1
    T[:,5:6] = (np.sin(np.cumsum(W[:,0:1]*stepvar,axis=2)+off_rh+offsetvar)>thesvar+np.cos(W[:,3:4])).astype(np.float)*2-1
    T[:,6:7] = (np.sin(np.cumsum(W[:,0:1]*stepvar,axis=2)+off_rt+offsetvar)>thesvar+np.cos(W[:,4:5])).astype(np.float)*2-1

    mvel = np.sqrt(np.sum(T[:,:2]**2, axis=1))
    for i in range(T.shape[0]):
        T[i,:,mvel[i]<0.75] = 1

    #############

    network_first, network_second, network = create_network(batchsize=T.shape[0], window=T.shape[2], hidden=T.shape[1])
    network_func = theano.function([input], network(input), allow_input_downcast=True)
#    pydotprint(network_func, 'crowd_network.png')

    start = time.clock()
    X = network_func(T)
    X = (X * preprocess['Xstd']) + preprocess['Xmean']
    Xtail = (T * preprocess['Xstd'][:,-7:]) + preprocess['Xmean'][:,-7:]
    X = constrain(X, network_second[0], network_second[1], preprocess, multiconstraint(
        foot_sliding(Xtail[:,-4:]),
        joint_lengths(),
        trajectory(Xtail[:,:3])), alpha=0.01, iterations=10)
    X[:,-7:] = Xtail

    #############

    animation_plot([X[0:1,:,:200], X[10:11,:,:200], X[20:21,:,:200]], interval=15.15)

    X = np.swapaxes(X, 1, 2)

    joints = X[:,:,:-7].reshape((X.shape[0], X.shape[1], -1, 3))
    joints = -Quaternions(data[scene+'_rot'][:,cstart:cend])[:,:,np.newaxis] * joints
    joints[:,:,:,0] += data[scene+'_pos'][:,cstart:cend][:,:,np.newaxis][:,:,:,0]
    joints[:,:,:,2] += data[scene+'_pos'][:,cstart:cend][:,:,np.newaxis][:,:,:,2]

    #np.savez_compressed('./videos/crowd/'+scene+'.npz', X=joints)



