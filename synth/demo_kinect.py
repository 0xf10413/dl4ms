#!/usr/bin/env python3
import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from network import create_core
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

rng = np.random.RandomState(23455)

Xkinect = np.load('../data/processed/data_edin_kinect.npz')['clips']
Xkinect = np.swapaxes(Xkinect, 1, 2).astype(theano.config.floatX)
Xsens = np.load('../data/processed/data_edin_xsens.npz')['clips']
Xsens = np.swapaxes(Xsens, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')
Xkinect = (Xkinect - preprocess['Xmean']) / preprocess['Xstd']
Xsens = (Xsens - preprocess['Xmean']) / preprocess['Xstd']

batchsize = 1
window = Xkinect.shape[2]*4

Xkinect = theano.shared(Xkinect, borrow=True)
Xsens = theano.shared(Xsens, borrow=True)

network = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2)
network.load(np.load('network_core.npz'))

from AnimationPlot import animation_plot

index = Xkinect.shape[0].eval()-82
#index = rng.randint(Xkinect.shape[0].eval())
print(index)

Xsensclip = T.concatenate([
    Xsens[index+0:index+1,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+1:index+2,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+2:index+3,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+3:index+4,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+4:index+5,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+5:index+6,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+6:index+7,:,Xsens.shape[2]//4:-Xsens.shape[2]//4],
    Xsens[index+7:index+8,:,Xsens.shape[2]//4:-Xsens.shape[2]//4]], axis=2)

Xkinectclip = T.concatenate([
    Xkinect[index+0:index+1,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+1:index+2,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+2:index+3,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+3:index+4,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+4:index+5,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+5:index+6,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+6:index+7,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4],
    Xkinect[index+7:index+8,:,Xkinect.shape[2]//4:-Xkinect.shape[2]//4]], axis=2)

Xorgi = np.array(Xsensclip.eval())
Xnois = np.array(Xkinectclip.eval())
Xrecn = np.array(network(Xnois).eval())    

Xorgi = (Xorgi * preprocess['Xstd']) + preprocess['Xmean']
Xnois = (Xnois * preprocess['Xstd']) + preprocess['Xmean']
Xrecn = (Xrecn * preprocess['Xstd']) + preprocess['Xmean']

Xrecn = constrain(Xrecn, network[0], network[1], preprocess, multiconstraint(
    foot_sliding(Xorgi[:,-4:].copy()),
    joint_lengths(),
    trajectory(Xorgi[:,-7:-4].copy())), alpha=0.01, iterations=100)

   
animation_plot([Xnois, Xrecn, Xorgi], interval=15.15)
    
