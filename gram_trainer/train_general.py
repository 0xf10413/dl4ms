#!/usr/bin/env python3
import sys, os
import numpy as np
import scipy.io as io
from sklearn.preprocessing import normalize
import theano
import theano.tensor as T
import theano.tensor.fft as TF

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

"""
Classe les clips selon le mouvement ou le style
Utilise une BDD répartie aléatoirement (par "tas" de style ou de mouvement)
Passe par un SLP
"""
if len(sys.argv) < 2:
    raise ValueError("Missing commandline argument")

sys.path.append('../nn')
sys.path.append('../synth')

from network import create_core
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

rng = np.random.RandomState(23455)

print("Loading data")
Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
Xedin_punching = np.load('../data/processed/data_edin_punching.npz')['clips']
Xedin_locomotion = np.load('../data/processed/data_edin_locomotion.npz')['clips']
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
Xedin_punching = np.swapaxes(Xedin_punching, 1, 2).astype(theano.config.floatX)
Xedin_locomotion = np.swapaxes(Xedin_locomotion, 1, 2).astype(theano.config.floatX)

preprocess = np.load('../synth/preprocess_core.npz')

Xstyletransfer = (Xstyletransfer - preprocess['Xmean']) / preprocess['Xstd']
Xedin_punching = (Xedin_punching - preprocess['Xmean']) / preprocess['Xstd']

def create_network(batchsize, window):
    network = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2, rng=rng)
    network.load(np.load('../synth/network_core.npz'))
    return network

pairings = [
   (i, Xstyletransfer) for i in range(len(Xstyletransfer))
]

timelen = 240
filename = 'all_grams_general.npz'
print("Done loading data")

grams = dict()

def storify(S0):
    """
    Transforme le tableau de mouvements selon sys.argv[1]
    Attendu : un np.array de dim 3
    """
    G = None
    H = None
    arg = sys.argv[1]
    if type(S0) is not type(np.array(tuple())):
        raise ValueError("Can only deal with np.array, got {}".
                format(type(S0)))
    if S0.ndim != 3:
        raise ValueError(
                "Can only deal with np.array of dim 3, got {} (shape {})".
                format(S0.ndim, S0.shape))

    if "caché" in arg:
        H = through_encoder(S0)
    elif "orig" in arg or "direct" in arg:
        H = S0
    else:
        raise ValueError("Invalid argument (could not find either" +
                "'caché' or 'orig' or 'direct') in " + arg )

    if "gram" in arg:
        G = to_gram(H)
    elif "nothing" in arg:
        G = H
    elif "fourier" in arg:
        G = to_fft(H)
        if "abs" in arg:
            realG = G[...,0].copy()
            imagG = G[...,1].copy()
            absG = np.abs(realG + imagG*1j)
            angG = np.angle(realG + imagG*1j)
            if "phase" in arg:
                G = np.concatenate((absG, angG))
            else:
                G = absG
    else:
        raise ValueError("Invalid argument (could not find either" +
                "'gram' or 'fourier' or 'nothing') in " + arg)
    return G

def to_gram():
    X = T.tensor3()
    G = T.sum(X.dimshuffle(0,'x',1,2) * X.dimshuffle(0,1,'x',2), axis=3)
    return theano.function([X], G)
to_gram = to_gram()

network = create_network(1, timelen)

def through_encoder():
    X = T.tensor3()
    H = network[0](X)
    return theano.function([X], H)
through_encoder = through_encoder()

def to_fft():
    H = T.tensor3()
    G = TF.rfft(H)
    return theano.function([H], G)
to_fft = to_fft()

if not os.path.isfile(filename):
    print("Generating {}".format(filename))
    def gram_matrix(X):
        return T.sum(X.dimshuffle(0,'x',1,2) * X.dimshuffle(0,1,'x',2), axis=3)
    def implot(X,title=""):
        plt.imshow(X)
        plt.colorbar()
        plt.title(title)
        plt.show()

    for i, db in pairings:
        S = [db[i:i+1]]
        assert S[0].shape[2] == timelen, "Wrong time shape"
        grams[str(i)] = storify(S[0])
        print("Done with matrix {}".format(i))

    np.savez(filename, **grams)
    print("Saved {}".format(filename))

C = np.load(filename)

def extract_and_cat(C, indexes,ctor=str):
    """
    Extrait les matrices d'indice dans indexes de C,
    puis les applatit et les concatène
    """
    matrices = [C[ctor(i)].flatten() for i in indexes]
    output = np.stack(matrices)
    output = normalize(output)
    return output

"""
Création de l'ensemble de données d'entraînement/test
et des targets correspondantes, pour les mouvements
"""
learning_filename = 'general_motions.npz'
if not os.path.isfile(learning_filename):
    database = extract_and_cat(C, range(559))
    training = None
    training_target = None
    testing = None
    testing_target = None
    cl = styletransfer_classes
    for mo in range(len(styletransfer_motions)):
        where = np.where(cl[:,0] == mo)[0]
        if not len(where):
            continue
        np.random.shuffle(where)
        stop = 3*len(where)//4
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
training_motion, testing_motion = D['training'], D['testing']
training_target_motion, testing_target_motion = D['training_target'], D['testing_target']

"""
Création de l'ensemble de données d'entraînement/test
et des targets correspondantes, pour les styles
"""
learning_filename = 'general_styles.npz'
if not os.path.isfile(learning_filename):
    database = extract_and_cat(C, range(559))
    training = None
    training_target = None
    testing = None
    testing_target = None
    cl = styletransfer_classes
    for st in range(len(styletransfer_styles)):
        where = np.where(cl[:,0] == st)[0]
        if not len(where):
            continue
        np.random.shuffle(where)
        stop = 3*len(where)//4
        if training is None:
            training = database[where[:stop]]
            training_target = cl[where[:stop]][:,1]
        else:
            training = np.concatenate((training,
                database[where[:stop]]))
            training_target = np.concatenate((training_target,
                cl[where[:stop]][:,1]))
        if testing is None:
            testing = database[where[stop:]]
            testing_target = cl[where[stop:]][:,1]
        else:
            testing = np.concatenate((testing,
                database[where[stop:]]))
            testing_target = np.concatenate((testing_target,
                cl[where[stop:]][:,1]))

        print("Done with style {}".format(styletransfer_styles[st]))
    np.savez(learning_filename, training=training, testing=testing,
            training_target=training_target, testing_target=testing_target)

D = np.load(learning_filename)
training_style, testing_style = D['training'], D['testing']
training_target_style, testing_target_style = D['training_target'], D['testing_target']

class SLP(nn.Module):
    """
    Perceptron à une seule couche
    La sortie est passée à travers un softlogmax
    """
    def __init__(self,output_size):
        super(SLP, self).__init__()
        self.fc1 = nn.Linear(training_style.shape[1], output_size)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self,x):
        x = self.fc1(x)
        x = self.logsoftmax(x)
        return x

#######################################
"""
Premier apprentissage : mouvements
"""
#######################################

input = Variable(torch.from_numpy(training_motion))
target = Variable(torch.from_numpy(training_target_motion))

number_of_classes = 8
net = SLP(number_of_classes)
print("Training with a SLP. Réseau utilisé :", net)
crit = nn.NLLLoss()
#optimizer = optim.SGD(net.parameters(), lr=.5)
#optimizer = optim.SGD(net.parameters(), lr=.01, momentum=.5)
optimizer = optim.Adam(net.parameters(), lr=.1)

"""
Entraînement effectif du réseau
"""
trained_file_move = "nn_motion_trained_general.npz"
if not os.path.isfile(trained_file_move):
    max_epoch = 5+1
    losses = np.zeros((max_epoch,))
    for epoch in range(max_epoch):
        optimizer.zero_grad()

        output = net(input.float())
        loss = crit(output, target)
        loss.backward()

        optimizer.step()
        #if epoch % (max_epoch // 10) == 0:
        print("Ending epoch {}, loss was {}".format(epoch, loss.data[0]))
        losses[epoch] = loss.data[0]
    print("Now saving...")
    torch.save(net.state_dict(), trained_file_move)
    plt.plot(losses)
    plt.title("Losses per epoch")
    #plt.show()


"""
Évaluation des résultats
"""
net.load_state_dict(torch.load(trained_file_move))
input = Variable(torch.from_numpy(testing_motion))
target = Variable(torch.from_numpy(testing_target_motion))
output = net(input.float())
confusion_matrix = np.zeros((8,8), dtype=np.int32)

for i in range(len(output)):
    predict = np.argmax(output.data[i].numpy())
    truth = testing_target_motion[i]
    confusion_matrix[truth, predict] += 1
    if predict != truth:
        print("{} : guessed {}, was {}".format(i,
            styletransfer_motions[predict],
            styletransfer_motions[truth]))

with open('confusion_matrix_general.csv', 'w+') as f:
    f.write("↓ Truth/Predicted →,")
    for cl in styletransfer_motions:
        f.write(cl + ',')
    f.write('\n')
    for line in range(8):
        f.write(styletransfer_motions[line] + ',')
        for col in range(8):
            f.write(str(confusion_matrix[line,col]))
            f.write(',')
        f.write('\n')

print("Confusion matrix written at confusion_matrix_general.csv")

accuracy = sum([confusion_matrix[i,i] for i in range(8)])/np.sum(confusion_matrix)

print("Accuracy is {:.2f}%".format(accuracy*100))

#######################################
"""
Second apprentissage : styles
"""
#######################################
print("Now for style learning")

input = Variable(torch.from_numpy(training_style))
target = Variable(torch.from_numpy(training_target_style))

number_of_classes = 8
net = SLP(number_of_classes)
print("Training with a SLP. Réseau utilisé :", net)
crit = nn.NLLLoss()
#optimizer = optim.SGD(net.parameters(), lr=.5)
#optimizer = optim.SGD(net.parameters(), lr=.01, momentum=.5)
optimizer = optim.Adam(net.parameters(), lr=.1)

"""
Entraînement effectif du réseau
"""
trained_file_style = "nn_style_trained_general.npz"
if not os.path.isfile(trained_file_style):
    max_epoch = 5+1
    losses = np.zeros((max_epoch,))
    for epoch in range(max_epoch):
        optimizer.zero_grad()

        output = net(input.float())
        loss = crit(output, target)
        loss.backward()

        optimizer.step()
        #if epoch % (max_epoch // 10) == 0:
        print("Ending epoch {}, loss was {}".format(epoch, loss.data[0]))
        losses[epoch] = loss.data[0]
    print("Now saving...")
    torch.save(net.state_dict(), trained_file_style)
    plt.plot(losses)
    plt.title("Losses per epoch")
    #plt.show()


"""
Évaluation des résultats
"""
net.load_state_dict(torch.load(trained_file_style))
input = Variable(torch.from_numpy(testing_style))
target = Variable(torch.from_numpy(testing_target_style))
output = net(input.float())
confusion_matrix = np.zeros((8,8), dtype=np.int32)

for i in range(len(output)):
    predict = np.argmax(output.data[i].numpy())
    truth = testing_target_style[i]
    truth_motion = testing_target_style[i]
    confusion_matrix[truth, predict] += 1
    if predict != truth:
        print("{} : guessed {}, was {}+{}".format(i,
            styletransfer_styles[predict],
            styletransfer_styles[truth],
            styletransfer_motions[truth_motion]))

with open('confusion_matrix2_general.csv', 'w+') as f:
    f.write("↓ Truth/Predicted →,")
    for cl in styletransfer_styles:
        f.write(cl + ',')
    f.write('\n')
    for line in range(8):
        f.write(styletransfer_styles[line] + ',')
        for col in range(8):
            f.write(str(confusion_matrix[line,col]))
            f.write(',')
        f.write('\n')

print("Confusion matrix written at confusion_matrix2_general.csv")

accuracy = sum([confusion_matrix[i,i] for i in range(8)])/np.sum(confusion_matrix)

print("Accuracy is {:.2f}%".format(accuracy*100))

#######################################
"""
Test : identification des mouvements tirés d'une autre BDD
"""
#######################################
net = SLP(number_of_classes)
net.load_state_dict(torch.load(trained_file_move))

# Edin punching, move
def get_target_move(ind):
    if ind in range(0, 30):
        return "fight_pose"
    if ind in range(30, 57):
        return "punching"
    if ind in range(57, 85):
        return "kicking"
    if ind in range(85, 95):
        return "elbow"
    return "something else"
# Edin punching, style
def get_target_style(_):
    return "neutral"
get_target = get_target_style
db = Xedin_punching
indexes = range(1,200)

# Edin locomotion
def get_target_motion(_):
    return "walking"
get_target = get_target_motion
db = Xedin_locomotion
indexes = range(1,200)

## Already "Gramified" data
#indexes = [(230,321),(234, 39),(220,148),(225,-41)]
#db = np.load("../synth/X_styletransfer_{}_{}.npz".
#        format(*indexes[0]))['Xtrsf']
#for i,j in indexes[1:]:
#    db2 = np.load("../synth/X_styletransfer_{}_{}.npz".
#        format(i,j))['Xtrsf']
#    db = np.concatenate((db, db2))
#db = db.astype(theano.config.floatX)
#db = (db - preprocess['Xmean']) /preprocess['Xstd']
#
#db = np.concatenate(np.split(db, 2, axis=2), axis=0)
#indexes = range(db.shape[0])
#def get_target_style(index):
#    if i <= 1:
#        return "old"
#    if i <= 3:
#        return "angry"
#    if i <= 5:
#        return "depressed"
#    if i <= 7:
#        return "strutting"
#    return "something else"
#get_target = get_target_motion

to_extract = [storify(db[i:i+1]) for i in range(db.shape[0])]
A = extract_and_cat(to_extract,
    indexes, ctor=int)
output = net(Variable(torch.from_numpy(A)).float()).data.numpy()

for i, cl in enumerate(np.argmax(output, axis=1)):
    #print("Thought #{} was {}, was actually {}".
    print("{},{}".
            format(styletransfer_motions[cl], get_target(i)))
