#!/usr/bin/env python3
import sys, os
import numpy as np
import scipy.io as io
from sklearn.preprocessing import normalize
import theano
import theano.tensor as T

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

"""
Tente de classifier les clips selon le mouvement
"""

sys.path.append('../nn')
sys.path.append('../synth')

from network import create_core
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

rng = np.random.RandomState(23455)

print("Loading data")
Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
# Format : motion x style
styletransfer_classes = np.load('../data/processed/data_styletransfer.npz')['classes']
styletransfer_styles = [
    'angry', 'childlike', 'depressed', 'neutral',
    'old', 'proud', 'sexy', 'strutting']
styletransfer_motions = [
    'fast_punching', 'fast_walking', 'jumping',
    'kicking', 'normal_walking', 'punching',
    'running', 'transitions']


Xstyletransfer = np.swapaxes(Xstyletransfer, 1, 2).astype(theano.config.floatX)

preprocess = np.load('../synth/preprocess_core.npz')

Xstyletransfer = (Xstyletransfer - preprocess['Xmean']) / preprocess['Xstd']

def create_network(batchsize, window):
    network = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2, rng=rng)
    network.load(np.load('../synth/network_core.npz'))
    return network

pairings = [
   (i, Xstyletransfer) for i in range(len(Xstyletransfer))
]
timelen = 240
filename = 'all_grams.npz'
print("Done loading data")

grams = dict()
if not os.path.isfile(filename):
    network = create_network(1, timelen)
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

class SLP(nn.Module):
    """
    Perceptron à une seule couche
    """
    def __init__(self,output_size):
        super(SLP, self).__init__()
        self.fc1 = nn.Linear(256*256, output_size)
        #self.fc2 = nn.Linear(256, output_size)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self,x):
        x = self.fc1(x)
        #x = self.fc2(x)
        x = self.logsoftmax(x)
        return x


# Entraînement sur deux mouvements différents pour un même style (normal)
# fast_punching et fast_walking
def extract_and_cat(C, indexes):
    """
    Extrait les matrices d'indice dans indexes de C,
    puis les applatit et les concatène
    """
    matrices = [C[str(i)].flatten() for i in indexes]
    output = np.stack(matrices)
    output = normalize(output)
    return output


learning_filename = 'motions_learning.npz'
if not os.path.isfile(learning_filename):
    database = extract_and_cat(C, range(559))
    training = None
    training_target = None
    testing = None
    testing_target = None
    cl = styletransfer_classes
    for mo in range(len(styletransfer_motions)):
        for st in range(len(styletransfer_styles)):
            where = np.where(
                    np.logical_and(cl[:,0] == mo, cl[:,1] == st))[0]
            if not len(where):
                continue
            for i in np.nditer(where):
                np.random.shuffle(where)
                stop = len(where)//2
                if training is None:
                    training = database[where[:stop]]
                    training_target = cl[where[:stop]][:,0]
                else:
                    training = np.concatenate((training,
                        database[where[:stop]]))
                    training_target = np.concatenate((training_target,
                        cl[where[:stop]][:,0]))
                if testing is None:
                    testing = database[where[stop:]]
                    testing_target = cl[where[stop:]][:,0]
                else:
                    testing = np.concatenate((testing,
                        database[where[stop:]]))
                    testing_target = np.concatenate((testing_target,
                        cl[where[stop:]][:,0]))

        print("Done with motion {}".format(styletransfer_motions[mo]))
    np.savez(learning_filename, training=training, testing=testing,
            training_target=training_target, testing_target=testing_target)

D = np.load(learning_filename)
training, testing = D['training'], D['testing']
training_target, testing_target = D['training_target'], D['testing_target']

input = Variable(torch.from_numpy(training))
target = Variable(torch.from_numpy(training_target))

number_of_classes = 8
net = SLP(number_of_classes)
print("Training with a SLP. Réseau utilisé :", net)
crit = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=.5)
#optimizer = optim.SGD(net.parameters(), lr=.01, momentum=.5)
#optimizer = optim.Adam(net.parameters(), lr=.001)

trained_file = "nn_motion_trained.npz"
if not os.path.isfile(trained_file):
    max_epoch = 1000+1
    losses = np.zeros((max_epoch,))
    for epoch in range(max_epoch):
        output = net(input)
        loss = crit(output, target)
        loss.backward()
        optimizer.step()
        #if epoch % (max_epoch // 10) == 0:
        print("Ending epoch {}, loss was {}".format(epoch, loss.data[0]))
        losses[epoch] = loss.data[0]
    print("Now saving...")
    torch.save(net.state_dict(), trained_file)
    plt.plot(losses)
    plt.title("Losses per epoch")
    plt.show()

net.load_state_dict(torch.load(trained_file))
print("Check :")
a = net(input)
print(a)

