import sys
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

sys.path.append('../nn')

from network import create_core
from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

rng = np.random.RandomState(23455)

Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
Xhdm05 = np.load('../data/processed/data_hdm05.npz')['clips']
Xedin_locomotion = np.load('../data/processed/data_edin_locomotion.npz')['clips']
Xedin_misc = np.load('../data/processed/data_edin_misc.npz')['clips']

Xstyletransfer = np.swapaxes(Xstyletransfer, 1, 2).astype(theano.config.floatX)
Xhdm05 = np.swapaxes(Xhdm05, 1, 2).astype(theano.config.floatX)
Xedin_locomotion = np.swapaxes(Xedin_locomotion, 1, 2).astype(theano.config.floatX)
Xedin_misc = np.swapaxes(Xedin_misc, 1, 2).astype(theano.config.floatX)

preprocess = np.load('preprocess_core.npz')

Xstyletransfer = (Xstyletransfer - preprocess['Xmean']) / preprocess['Xstd']
Xhdm05 = (Xhdm05 - preprocess['Xmean']) / preprocess['Xstd']
Xedin_locomotion = (Xedin_locomotion - preprocess['Xmean']) / preprocess['Xstd']
Xedin_misc = (Xedin_misc - preprocess['Xmean']) / preprocess['Xstd']

def create_network(batchsize, window):
    network = create_core(batchsize=batchsize, window=window, dropout=0.0, depooler=lambda x,**kw: x/2, rng=rng)
    network.load(np.load('network_core.npz'))
    return network

pairings = [
   (245, Xedin_locomotion, 100, Xedin_misc,  0.1),
   (242, Xedin_locomotion, 110, Xedin_misc,  0.1),
   (238, Xedin_locomotion,  90, Xedin_misc,  0.1),

   ( 53, Xedin_locomotion,  50, Xedin_misc,  0.1),
   ( 52, Xedin_locomotion,  71, Xedin_misc,  0.1),
   ( 15, Xedin_locomotion,  30, Xedin_misc,  0.1),

   (234, Xedin_locomotion,  39, Xstyletransfer,  0.1),
   (230, Xedin_locomotion, 321, Xstyletransfer,  0.1),
   (225, Xedin_locomotion,  -41, Xstyletransfer, 0.1),

   (220, Xedin_locomotion,  148, Xstyletransfer,  0.1),
]

for content_clip, content_database, style_clip, style_database, style_amount in pairings:

    S = style_database[style_clip:style_clip+1]
    C = np.concatenate([
        content_database[content_clip+0:content_clip+1,:,S.shape[2]//4:-S.shape[2]//4],
        content_database[content_clip+1:content_clip+2,:,S.shape[2]//4:-S.shape[2]//4],
        content_database[content_clip+2:content_clip+3,:,S.shape[2]//4:-S.shape[2]//4],
        content_database[content_clip+3:content_clip+4,:,S.shape[2]//4:-S.shape[2]//4]], axis=2)

    network_S = create_network(1, S.shape[2])
    network_C = create_network(1, C.shape[2])

    def gram_matrix(X):
        return T.sum(X.dimshuffle(0,'x',1,2) * X.dimshuffle(0,1,'x',2), axis=3)

    G = np.array(gram_matrix(network_S[0](S)).eval())

    C_shape = C.shape
    N_bound = np.sqrt(6. / (C_shape[1] + C_shape[2]))
    N = (np.asarray(rng.uniform(low=-N_bound, high=N_bound, size=C_shape),
        dtype=theano.config.floatX) * preprocess['Xstd']) + preprocess['Xmean']

    def style_transfer(H, V):
        s, c =  style_amount, 1.0
        s, c = s / (s + c), c / (s + c)
        return s * T.mean((gram_matrix(H) - G)**2) + c * T.mean((H - network_C[0](C))**2)

    Xstyl = (S * preprocess['Xstd']) + preprocess['Xmean']
    Xcntn = (C * preprocess['Xstd']) + preprocess['Xmean']
    Xtrsf = N
    Xtrsf = constrain(Xtrsf, network_C[0], network_C[1], preprocess, style_transfer, iterations=250, alpha=0.01)

    Xtrsfvel = np.mean(np.sqrt(Xtrsf[:,-7:-6]**2 + Xtrsf[:,-6:-5]**2), axis=2)[:,:,np.newaxis]
    Xcntnvel = np.mean(np.sqrt(Xcntn[:,-7:-6]**2 + Xcntn[:,-6:-5]**2), axis=2)[:,:,np.newaxis]

    Xtail = Xtrsfvel * (Xcntn[:,-7:] / Xcntnvel)
    Xtail[:,-5:] = Xcntn[:,-5:]

    Xtrsf = constrain(Xtrsf, network_C[0], network_C[1], preprocess, multiconstraint(
        foot_sliding(Xtail[:,-4:]),
        joint_lengths(),
        trajectory(Xtail[:,:3])), alpha=0.01, iterations=100)
    Xtrsf[:,-7:] = Xtail

    Xstyl = np.concatenate([Xstyl, Xstyl], axis=2)

    from AnimationPlot import animation_plot

    animation_plot([Xstyl, Xcntn, Xtrsf], interval=15.15)

