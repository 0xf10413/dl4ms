#!/usr/bin/env python3
import sys, os
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
"""
Affiche quelques matrices de Gram
"""

sys.path.append('../nn')
sys.path.append('../synth')

from network import create_core
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

rng = np.random.RandomState(23455)

print("Loading data")
Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
Xhdm05 = np.load('../data/processed/data_hdm05.npz')['clips']
Xedin_locomotion = np.load('../data/processed/data_edin_locomotion.npz')['clips']
Xedin_misc = np.load('../data/processed/data_edin_misc.npz')['clips']
Xedin_punch = np.load('../data/processed/data_edin_punching.npz')['clips']

Xstyletransfer = np.swapaxes(Xstyletransfer, 1, 2).astype(theano.config.floatX)
Xhdm05 = np.swapaxes(Xhdm05, 1, 2).astype(theano.config.floatX)
Xedin_locomotion = np.swapaxes(Xedin_locomotion, 1, 2).astype(theano.config.floatX)
Xedin_misc = np.swapaxes(Xedin_misc, 1, 2).astype(theano.config.floatX)
Xedin_punch = np.swapaxes(Xedin_punch, 1, 2).astype(theano.config.floatX)

preprocess = np.load('../synth/preprocess_core.npz')

Xstyletransfer = (Xstyletransfer - preprocess['Xmean']) / preprocess['Xstd']
Xhdm05 = (Xhdm05 - preprocess['Xmean']) / preprocess['Xstd']
Xedin_locomotion = (Xedin_locomotion - preprocess['Xmean']) / preprocess['Xstd']
Xedin_misc = (Xedin_misc - preprocess['Xmean']) / preprocess['Xstd']
Xedin_punch = (Xedin_punch - preprocess['Xmean']) / preprocess['Xstd']

def create_network(batchsize, window):
    network = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2, rng=rng)
    network.load(np.load('../synth/network_core.npz'))
    return network

pairings = [
   #(i, Xstyletransfer) for i in (241, 242, 243, 259 ) # walk
   #(i, Xstyletransfer) for i in (241, 233, 264, 272)# walk, kick, punch, running
   #(i, Xstyletransfer) for i in (264, 58, 337)# punch neutral, angry, old
   (i, Xstyletransfer) for i in (241, 39, 317) # walk neutral, angry, old
   #(i, Xstyletransfer) for i in range(len(Xstyletransfer)) # all
]
timelen = 240
filename = 'grams.npz'
print("Done loading data")

network = create_network(1, timelen)
grams = dict()
if not os.path.isfile(filename):
    print("Generating {}".format(filename))
    for i, db in pairings:
        S = [db[i:i+1]]
        assert S[0].shape[2] == timelen, "Wrong time shape"

        def gram_matrix(X):
            return T.sum(X.dimshuffle(0,'x',1,2) * X.dimshuffle(0,1,'x',2), axis=3)
        def implot(X,title=""):
            plt.imshow(X)
            plt.colorbar()
            plt.title(title)
            plt.show()
        H = network[0](S[0])
        G = np.array(gram_matrix(H).eval())
        grams[str(i)] = G
        print("Done with matrix {}".format(i))

    np.savez(filename, **grams)
    print("Saved {}".format(filename))

C = np.load(filename)
indexes = list(C)
for i in range(len(indexes)):
    for j in range(i):
        print("norm2 between {} and {} (gram) : {}".format(
            indexes[i], indexes[j],
            np.mean((C[indexes[i]]-C[indexes[j]])**2)
            ))

for i in range(len(indexes)):
    #plt.subplot(4, len(indexes)//4+1, i+1, aspect="equal")
    plt.figure()
    to_draw = C[indexes[i]][0]
    plt.imshow(to_draw,cmap=plt.get_cmap('coolwarm'))
    plt.title(indexes[i])
    plt.xticks([])
    plt.yticks([])

    #plt.hist(to_draw)
    #plt.title(indexes[i])
    #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
#plt.tight_layout()
plt.show()
