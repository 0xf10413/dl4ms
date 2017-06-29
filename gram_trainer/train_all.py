#!/usr/bin/env python3
"""
Fonctions d'entraînement pour la classification des mouvements
ou des styles. Fonctionne aussi bien avec des réseaux de neurones
basés sur pytorch que des classificateurs générisques issus de
scikit
"""
import os
import warnings
import sys
import pickle

import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import theano
import theano.tensor as T
import theano.tensor.fft as TF
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

sys.path.append('../nn')
sys.path.append('../synth')
from network import create_core

########################
## Modèles génériques ##
########################
class SLP(nn.Module):
    """
    Perceptron à une seule couche
    La sortie est passée à travers un softlogmax
    """
    def __init__(self, input_size=1, output_size=1):
        super(SLP, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self,x):
        x = self.fc1(x)
        x = self.logsoftmax(x)
        return x

    def fit(self, X, y):
        """
        Apprend sur la bdd X et l'output attendu y
        """
        input = Variable(torch.from_numpy(X))
        target = Variable(torch.from_numpy(y))
        crit = nn.NLLLoss()
        optimizer = optim.Adam(self.parameters(), lr=.001)
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            output = self(input.float())
            loss = crit(output, target)
            loss.backward()
            optimizer.step()
            print("Ending epoch {}, loss was {:e}".
                    format(epoch, loss.data[0]), end='\r')
        print()

    def predict(self, X):
        input = Variable(torch.from_numpy(X).float())
        output = self(input)
        predict = np.argmax(output.data.numpy(), axis=1)
        return predict

    def save(self, filename):
        torch.save(self, filename)

    @staticmethod
    def load(file):
        return torch.load(file)

class SVM(object):
    def __init__(self):
        self.param_grid = [
        { 'C': [1e-2, 0.1, 1], 'kernel':['rbf'],
            'gamma' : [1e-2, 1e-1, 1, 10, 100, 1000],
        },
        { 'C': [1e-2, 0.1, 1], 'kernel':['linear'],
        },
        ]
        self.grid_search = GridSearchCV(svm.SVC(), self.param_grid, n_jobs=12, verbose=0)

    def fit(self, X, y):
        self.grid_search.fit(X,y)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.grid_search, f)

    @staticmethod
    def load(filename):
        svm = None
        with open(filename, 'rb') as f:
            svm = pickle.load(f)
        return svm


####################
## Outil de cache ##
####################
class Cacher(object):
    def __init__(self, arg):
        """
        Utilise arg pour créer un dossier de cache unique
        """
        self.path = "cache/" + arg
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def retrieve(self, filename, how_to, format='npz'):
        """
        Récupère le fichier numpy si possible, le reconstruit sinon
        """
        filename = self.path + "/" + filename + '.' + format
        if not os.path.exists(filename):
            how_to(filename)
        else:
            print("(Cacher: already done, skipping)")
        if not os.path.exists(filename):
            raise RuntimeError("how_to did not build file as expected")
        if format == 'npz':
            return np.load(filename, 'r')
        return filename



#######################################
## Génération des fichiers d'exemple ##
#######################################
def generate_gram_samples(pairings, arg, filename):
    grams = dict()
    for n, (i, db) in enumerate(pairings):
        S = db[i:i+1]
        assert S.shape[2] == timelen, "Wrong time shape ({} != {})".format(S.shape[2], timelen)
        grams[str(i)] = storify(S, arg)
        print("{} : {}/{}".
                format(generate_gram_samples.__name__,
                    n, len(pairings)-1),
                end='\r')
    print()
    grams = extract_and_cat(grams, range(len(grams)))
    np.savez(filename, grams=grams)

def generate_normalized_gram(database, norm, filename):
    np.savez(filename, grams=norm(database))


def generate_motions_samples(database, filename):
    database = database['grams']
    training = None
    training_target = None
    testing = None
    testing_target = None
    cl = styletransfer_classes
    print("Done with motion ", end='')
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

        print(styletransfer_motions[mo], end=' ')
    print()
    np.savez(filename, training=training, testing=testing,
            training_target=training_target, testing_target=testing_target)
    return np.load(filename)

def generate_styles_samples(database, filename):
    database = database['grams']
    training = None
    training_target = None
    testing = None
    testing_target = None
    cl = styletransfer_classes
    print("Done with style ", end='')
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

        print(styletransfer_styles[st], end=' ')
    print()
    np.savez(filename, training=training, testing=testing,
            training_target=training_target, testing_target=testing_target)

def generate_slp(database, filename):
    D = database
    training, testing = D['training'], D['testing']
    training_target, testing_target = D['training_target'], D['testing_target']
    net = SLP(training.shape[1], number_of_classes)
    net.fit(training, training_target)
    net.save(filename)

def generate_svm(database, filename):
    D = database
    training, testing = D['training'], D['testing']
    training_target, testing_target = D['training_target'], D['testing_target']
    svm = SVM()
    svm.fit(training, training_target)
    svm.save(filename)

def generate_external_samples(database, filename):
    cl = np.zeros((len(database)))
    np.savez(filename, testing=database, testing_target=cl)



################################################
## Fonctions génériques traitement de l'input ##
################################################
timelen = 240
max_epochs = 1000
rng = np.random.RandomState(23455)
number_of_classes = 8
styletransfer_classes = np.load('../data/processed/data_styletransfer.npz')['classes']
styletransfer_styles = [
    'angry', 'childlike', 'depressed', 'neutral',
    'old', 'proud', 'sexy', 'strutting']
styletransfer_motions = [
    'fast_punching', 'fast_walking', 'jumping',
    'kicking', 'normal_walking', 'punching',
    'running', 'transitions']


def storify(S0, arg):
    """
    Transforme le tableau de mouvements S0 selon l'argument en cli arg.
    Attendu : un np.array de dim 3 : batchsize x features x time
    """
    if type(S0) is not type(np.array(tuple())):
        raise ValueError(
                "Can only deal with np.array, got {}".
                format(type(S0)))
    if S0.ndim != 3:
        raise ValueError(
                "Can only deal with np.array of dim 3, got {} (shape {})".
                format(S0.ndim, S0.shape))
    G = None
    H = None

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

def through_encoder():
    network = create_core(batchsize=1, window=timelen,
            dropout=0.0, depooler=lambda x,**kw: x/2, rng=rng)
    network.load(np.load('../synth/network_core.npz'))
    X = T.tensor3()
    H = network[0](X)
    return theano.function([X], H)
through_encoder = through_encoder()

def to_fft():
    H = T.tensor3()
    G = TF.rfft(H)
    return theano.function([H], G)
to_fft = to_fft()

def norm_unnorm(database):
    mean = np.mean(database, axis=0)
    std = np.std(database, axis=0)
    return lambda x: np.nan_to_num((x - mean)/std), lambda y: y*std + mean

def extract_and_cat(C, indexes, normalize=None, ctor=str):
    """
    Extrait les matrices d'indice dans indexes de C,
    puis les aplatit et les concatène
    """
    matrices = [C[ctor(i)].flatten() for i in indexes]
    output = np.stack(matrices)
    if normalize is not None:
        output = normalize(output)
    return output

def to_confusion_matrix(clf, database, number_of_classes=number_of_classes):
    """
    Construit la matrice de confusion à partir des prédiction et des
    vérités terrains données en paramètre, en supposant les classes
    numérotées de 0 à number_of_classes
    """
    input = database['testing']
    correction = database['testing_target']
    prediction = clf.predict(input)
    confusion_matrix = np.zeros((number_of_classes, number_of_classes))
    for i in range(len(prediction)):
        predict = prediction[i]
        truth = correction[i]
        confusion_matrix[predict, truth] += 1
    accuracy = sum([confusion_matrix[i,i] for i in range(8)])/np.sum(confusion_matrix)
    return confusion_matrix, accuracy

def to_nice_table(clf, database, classes):
    """
    Construit une table de résultats de prédiction de clf sur database
    en utilisant les noms de classes
    """
    input = database['testing']
    prediction = clf.predict(input)
    nice_table = [[cl,0] for cl in classes]
    for i in range(len(prediction)):
        predict = prediction[i]
        nice_table[predict][1] += 1
    return nice_table

def write_accuracy(clf, database, filename):
    """
    Ecrit l'accuracy dans un fichier
    """
    _, accuracy = to_confusion_matrix(clf, database)
    with open(filename, 'w') as f:
        print("{0:.1%}".format(accuracy), file=f)

def to_csv(clf, database, classes, expect, filename):
    """
    Construit une table de résultats de prédiction de clf sur database
    en utilisant les noms de classes. Met en format csv
    """
    input = database['testing']
    prediction = clf.predict(input)
    with open(filename, 'w') as f:
        print("prediction,expected", file=f)
        for i in range(len(prediction)):
            predict = classes[prediction[i]]
            truth = expect(i)
            print(predict, truth, sep=',', file=f)


if __name__ == "__main__":
    Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
    Xstyletransfer = np.swapaxes(Xstyletransfer, 1, 2).astype(theano.config.floatX)
    Xedin_punching = np.load('../data/processed/data_edin_punching.npz')['clips']
    Xedin_punching = np.swapaxes(Xedin_punching, 1, 2).astype(theano.config.floatX)
    Xedin_locomotion = np.load('../data/processed/data_edin_locomotion.npz')['clips']
    Xedin_locomotion = np.swapaxes(Xedin_locomotion, 1, 2).astype(theano.config.floatX)

    preprocess = np.load('../synth/preprocess_core.npz')
    Xstyletransfer = (Xstyletransfer - preprocess['Xmean']) / preprocess['Xstd']
    Xedin_punching = (Xedin_punching - preprocess['Xmean']) / preprocess['Xstd']
    Xedin_locomotion = (Xedin_locomotion - preprocess['Xmean']) / preprocess['Xstd']

    arg = sys.argv[1]
    pairings = [(i, Xstyletransfer) for i in range(len(Xstyletransfer))]
    how_to = lambda filename: generate_gram_samples(pairings, arg, filename)
    print("=> Fetching all samples")
    C = Cacher(arg).retrieve("grams", how_to)
    norm, unnorm = norm_unnorm(C['grams'])

    print("=> Normalizing samples")
    how_to = lambda filename: generate_normalized_gram(C['grams'], norm, filename)
    C = Cacher(arg).retrieve("grams_normalized", how_to)

    print("=> Generating motions samples")
    how_to = lambda filename: generate_motions_samples(C, filename)
    M = Cacher(arg).retrieve("motions", how_to)

    print("=> Generating styles samples")
    how_to = lambda filename: generate_styles_samples(C, filename)
    S = Cacher(arg).retrieve("styles", how_to)

    print("=> Training SLP on motion samples")
    how_to = lambda filename: generate_slp(M, filename)
    slp_motions = SLP.load(Cacher(arg).retrieve("slp_motions", how_to, format='torch'))
    how_to = lambda f: write_accuracy(slp_motions, M, f)
    Cacher(arg).retrieve("slp_motions_accuracy", how_to, format='txt')

    print("=> Training SLP on style samples")
    how_to = lambda filename: generate_slp(S, filename)
    slp_styles = SLP.load(Cacher(arg).retrieve("slp_styles", how_to, format='torch'))
    how_to = lambda f: write_accuracy(slp_styles, S, f)
    Cacher(arg).retrieve("slp_styles_accuracy", how_to, format='txt')

    print("=> Training SVM on motion samples")
    how_to = lambda filename: generate_svm(M, filename)
    svm_motions = SVM.load(Cacher(arg).retrieve("svm_motions", how_to, format='scikit'))
    how_to = lambda f: write_accuracy(svm_motions, M, f)
    Cacher(arg).retrieve("svm_motions_accuracy", how_to, format='txt')

    print("=> Training SVM on style samples")
    how_to = lambda filename: generate_svm(S, filename)
    svm_styles = SVM.load(Cacher(arg).retrieve("svm_styles", how_to, format='scikit'))
    how_to = lambda f: write_accuracy(svm_styles, S, f)
    Cacher(arg).retrieve("svm_styles_accuracy", how_to, format='txt')

    print("=> Now checking on edin_punching")
    print("=> Fetching new data")
    pairings = [(i, Xedin_punching) for i in range(100)]
    how_to = lambda filename: generate_gram_samples(pairings, arg, filename)
    C = Cacher(arg).retrieve("edin_punching", how_to)

    print("=> Normalizing new data")
    how_to = lambda filename: generate_normalized_gram(C['grams'], norm, filename)
    C = Cacher(arg).retrieve("edin_punching_normalized", how_to)

    print("= Finalizing samples")
    how_to = lambda filename: generate_external_samples(C['grams'], filename)
    C = Cacher(arg).retrieve("edin_punching_final", how_to)

    print("=> Getting nice table for SLP")
    print(to_nice_table(slp_styles, C, styletransfer_styles))
    print(to_nice_table(slp_motions, C, styletransfer_motions))
    print("=> Getting nice table for SVM")
    print(to_nice_table(svm_styles, C, styletransfer_styles))
    print(to_nice_table(svm_motions, C, styletransfer_motions))

    how_to = lambda clf, cls, exp: lambda f:to_csv(clf, C, cls, exp, f)
    def get_edin_punching_move(ind):
        if ind in range(0, 30):
            return "fight_pose"
        if ind in range(30, 57):
            return "punching"
        if ind in range(57, 85):
            return "kicking"
        if ind in range(85, 95):
            return "elbow"
        return "something else"
    def get_edin_punching_style(_):
        return "neutral"
    print(Cacher(arg).retrieve("edin_punching_svm_style", how_to(svm_styles,
         styletransfer_styles, get_edin_punching_style), format="csv"))
    print(Cacher(arg).retrieve("edin_punching_svm_motion", how_to(svm_motions,
        styletransfer_motions, get_edin_punching_move), format="csv"))
    print(Cacher(arg).retrieve("edin_punching_slp_style", how_to(slp_styles,
        styletransfer_styles, get_edin_punching_style), format="csv"))
    print(Cacher(arg).retrieve("edin_punching_slp_motion", how_to(slp_motions,
        styletransfer_motions, get_edin_punching_move), format="csv"))


    print("=> Now checking on edin_locomotion")
    print("=> Fetching new data")
    pairings = [(i, Xedin_locomotion) for i in range(100)]
    how_to = lambda filename: generate_gram_samples(pairings, arg, filename)
    C = Cacher(arg).retrieve("edin_locomotion", how_to)

    print("=> Normalizing new data")
    how_to = lambda filename: generate_normalized_gram(C['grams'], norm, filename)
    C = Cacher(arg).retrieve("edin_locomotion_normalized", how_to)

    print("=> Finalizing samples")
    how_to = lambda filename: generate_external_samples(C['grams'], filename)
    C = Cacher(arg).retrieve("edin_locomotion_final", how_to)

    how_to = lambda clf, cls, exp: lambda f:to_csv(clf, C, cls, exp, f)
    def get_edin_locomotion_move(_):
        return "walking"
    def get_edin_locomotion_style(_):
        return "neutral"
    print(Cacher(arg).retrieve("edin_locomotion_svm_style", how_to(svm_styles,
         styletransfer_styles, get_edin_locomotion_style), format="csv"))
    print(Cacher(arg).retrieve("edin_locomotion_svm_motion", how_to(svm_motions,
        styletransfer_motions, get_edin_locomotion_move), format="csv"))
    print(Cacher(arg).retrieve("edin_locomotion_slp_style", how_to(slp_styles,
        styletransfer_styles, get_edin_locomotion_style), format="csv"))
    print(Cacher(arg).retrieve("edin_locomotion_slp_motion", how_to(slp_motions,
        styletransfer_motions, get_edin_locomotion_move), format="csv"))

