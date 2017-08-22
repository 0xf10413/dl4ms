#!/usr/bin/env python3
"""
Implémentation des GANs sur la génération de mouvements humains
"""
import os
import sys
from copy import copy, deepcopy

import numpy as np
import scipy

import matplotlib.pyplot as plt
from matplotlib import animation

if __debug__:
    os.environ['THEANO_FLAGS'] = 'device=cpu'
else:
    os.environ['THEANO_FLAGS'] = 'device=cuda'

import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
from theano import config
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
import lasagne.layers as ll
import lasagne.nonlinearities as lnl
import lasagne.init as lin

sys.path.append("../synth")
sys.path.append("../nn")
from network import create_core
from AnimationPlot import animation_plot
from constraints import joint_lengths
from Network import Network

class FDL4MSLayer(ll.Layer):
    def __init__(self, incoming, batch_size, window=240, **kwargs):
        super().__init__(incoming, **kwargs)
        enc_dec = create_core_flo(
                batchsize=batch_size,
                window=window,
                dropout=0.0,
                depooler=lambda x,**kw: x/2
                )
        enc_dec.load(np.load('../synth/network_core_flo.npz'))
        self.enc = Network(enc_dec[0], enc_dec[1], enc_dec[2])
        self.dec = Network(enc_dec[3], enc_dec[4], enc_dec[5])

        X = T.tensor3()
        self.encoder = theano.function([X], self.enc(X))
        self.decoder = theano.function([X], self.dec(X))

    def get_output_for(self, input, **kwargs):
        return self.dec(input)

    def get_output_shape_for(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[1] == 256
        assert input_shape[2] == 30
        return (input_shape[0], 73, 240)

class DL4MSLayer(ll.Layer):
    def __init__(self, incoming, batch_size, window=240, **kwargs):
        super().__init__(incoming, **kwargs)
        enc_dec = create_core(
                batchsize=batch_size,
                window=window,
                dropout=0.0,
                depooler=lambda x,**kw: x/2
                )
        enc_dec.load(np.load('../synth/network_core.npz'))
        self.enc = enc_dec[0]
        self.dec = enc_dec[1]

        X = T.tensor3()
        self.encoder = theano.function([X], self.enc(X))
        self.decoder = theano.function([X], self.dec(X))

    def get_output_for(self, input, **kwargs):
        return self.dec(input)

    def get_output_shape_for(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[1] == 256
        assert input_shape[2] == 120
        return (input_shape[0], 73, 240)


class GramLayer(ll.Layer):
    def __init__(self, incoming, nonlinearity=None):
        super().__init__(incoming)
        if nonlinearity is None:
            self.nonlinearity = lnl.identity
        else:
            self.nonlinearity = nonlinearity

    def get_output_for(self, input, **kwargs):
        left = input.dimshuffle(0, 'x', 1, 2)
        right = input.dimshuffle(0, 1, 'x', 2)
        s = T.sum(left*right, axis=3)
        return self.nonlinearity(s)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1]*input_shape[1])


if __name__ == "__main__":

    rng = np.random.RandomState(43)
    srng = RandomStreams(seed=rng.randint(0, np.iinfo(np.uint32).max))
    lasagne.random.set_rng(rng)

    batch_size = 1
    filter_size = 25
    x_shape = (batch_size, 73, 240)
    #z_shape = (batch_size, 256)
    z_shape = x_shape

    print("...loading motion data")
    #Xedin_locomotion = np.load('../data/processed/data_edin_locomotion.npz')['clips']
    #Xhdm05 = np.load('../data/processed/data_hdm05.npz')['clips']
    #Xcmu = np.load('../data/processed/data_cmu.npz')['clips']
    #Xmhad = np.load('../data/processed/data_mhad.npz')['clips']
    Xstyletransfer = np.load('../data/processed/data_styletransfer.npz')['clips']
    #Xedin_xsens = np.load('../data/processed/data_edin_xsens.npz')['clips']
    #Xedin_misc = np.load('../data/processed/data_edin_misc.npz')['clips']
    #Xedin_punching = np.load('../data/processed/data_edin_punching.npz')['clips']

    styletransfer_classes = np.load('../data/processed/data_styletransfer.npz')['classes']
    styletransfer_styles = [
        'angry', 'childlike', 'depressed', 'neutral',
        'old', 'proud', 'sexy', 'strutting']
    styletransfer_motions = [
        'fast_punching', 'fast_walking', 'jumping',
        'kicking', 'normal_walking', 'punching',
        'running', 'transitions']

    #X_train = np.concatenate([Xcmu, Xhdm05, Xmhad, Xstyletransfer, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)
    #X_train = np.concatenate([Xedin_locomotion], axis=0)
    X_train = Xstyletransfer[0:74] # Colère
    X_train_rev = Xstyletransfer[74:139] # Enfant
    X_train = np.swapaxes(X_train, 1, 2).astype(theano.config.floatX)
    X_train_rev = np.swapaxes(X_train_rev, 1, 2).astype(theano.config.floatX)

    preprocess = np.load('../synth/preprocess_core.npz')
    denorm = lambda x : x * preprocess['Xstd'] + preprocess['Xmean']
    norm = lambda x : (x - preprocess['Xmean']) / preprocess['Xstd']

    X_train = norm(X_train)
    X_train_rev = norm(X_train_rev)

    def pick_x(nb_samples=batch_size):
        assert nb_samples < len(X_train), "Not enough samples in pick_x"
        I = np.arange(len(X_train))
        rng.shuffle(I)
        return X_train[I[:nb_samples]].astype(config.floatX)

    def pick_x_rev(nb_samples=batch_size):
        assert nb_samples < len(X_train_rev), "Not enough samples in pick_x"
        I = np.arange(len(X_train_rev))
        rng.shuffle(I)
        return X_train_rev[I[:nb_samples]].astype(config.floatX)

    z_mean = 0
    z_std = 10

    def pick_z_norm(nb_samples=batch_size):
        return rng.normal(
                z_mean,
                z_std,
                (nb_samples, z_shape[1]),
                ).astype(config.floatX)

    def pick_z_unif(nb_samples=batch_size):
        return rng.uniform(
                z_mean-z_std,
                z_mean+z_std,
                (nb_samples, z_shape[1]),
                ).astype(config.floatX)
    pick_z = pick_x_rev
    pick_z_rev = pick_x

    def f_test(in_ctor, in_sample, layer_or_expr):
        if not __debug__:
            print("(Warning : disabled)")
            return
        if not type(in_ctor) == type(T.tensor3()) and \
            not type(in_ctor) == type([]):
            bbb = in_ctor(dtype=config.floatX)
        else:
            bbb = in_ctor
        if isinstance(in_sample, list):
            iii = tuple(in_sample)
        else:
            iii = (in_sample,)
        if isinstance(layer_or_expr, ll.Layer):
            aaa = ll.get_output(layer_or_expr, inputs=bbb)
        else:
            aaa = layer_or_expr
        if type(bbb) == type(list()):
            fff = theano.function(bbb, aaa)
        else:
            fff = theano.function([bbb], aaa)
        print(fff(*iii).shape)


    print('...building generator')
    # Generator
    g = {}

    g['in'] = ll.InputLayer(
        shape=z_shape,
        )
    print("=> Now testing g-in", end=" ")
    f_test(T.tensor3, pick_z(), g['in'])

    g['fc1'] = ll.DenseLayer(
        g['in'],
        num_units=256*120,
        nonlinearity=None,
        W=lin.GlorotUniform(),
        b=lin.Constant(0),
    )
    print("=> Now testing g-fc1", end=" ")
    f_test(T.tensor3, pick_z(), g['fc1'])

    g['resh'] = ll.ReshapeLayer(
        g['fc1'],
        shape=(batch_size, 256, 120),
    )
    print("=> Now testing g-resh", end=" ")
    f_test(T.tensor3, pick_z(), g['resh'])

    #g['conv1'] = ll.Conv1DLayer(
    #        g['resh'],
    #        num_filters=256,
    #        filter_size=13,
    #        pad='same',
    #        nonlinearity=None,
    #        )
    #print("=> Now testing g-conv", end=" ")
    #f_test(T.tensor3, pick_z(), g['conv1'])

    g['dec'] = DL4MSLayer(
            g['resh'],
            batch_size=batch_size,
            )
    encode = g['dec'].encoder
    decode = g['dec'].decoder
    print("=> Now testing g-dec", end=" ")
    f_test(T.tensor3, pick_z(), g['dec'])

    g['out'] = g['dec']

    print("=> Now testing generator", end=" ")
    f_test(T.tensor3, pick_z(), g['out'])
    x = T.tensor3()
    generate = ll.get_output(g['out'], inputs=x)
    generate = theano.function([x], generate)

    g_rev = deepcopy(g)
    generate_rev = ll.get_output(g_rev['out'], inputs=x)
    generate_rev = theano.function([x], generate_rev)

    print('...building discriminator')
    # Discriminator
    d = {}

    d['in'] = ll.InputLayer(
        shape=x_shape,
        )

    d['gram'] = GramLayer(
            d['in'],
            nonlinearity=None,
            )

    d['h1'] = ll.DenseLayer(
        d['gram'],
        num_units=73*73,
        nonlinearity=lnl.tanh,
        W=lin.GlorotUniform(),
        b=lin.Constant(0)
    )

    d['h2'] = ll.DenseLayer(
            d['h1'],
            num_units=1,
            nonlinearity=lnl.sigmoid,
            W=lin.GlorotUniform(),
            b=lin.Constant(0)
            )

    d['out'] = d['h2']
    x = T.tensor3()
    discriminate = ll.get_output(d['out'], inputs=x)
    discriminate = theano.function([x], discriminate)

    d_rev = deepcopy(d)
    discriminate_rev = ll.get_output(d_rev['out'], inputs=x)
    discriminate_rev = theano.function([x], discriminate_rev)

    print("=> Now testing discriminor", end=" ")
    f_test(T.tensor3, pick_x(), d['out'])

    print('generator shape:')
    for l in ll.get_all_layers(g['out']):
        print(format(l.output_shape) + ' ' + format(l.__class__))

    print('discriminor shape:')
    for l in ll.get_all_layers(d['out']):
        print(format(l.output_shape) + ' ' + format(l.__class__))

## Direct
    z = T.tensor(dtype=config.floatX, broadcastable=(False, False, False))
    x = T.tensor(dtype=config.floatX, broadcastable=(False, False, False))

    x_fake = ll.get_output(g['out'], inputs=z)
    print("=> Now testing x_fake", end=" ")
    f_test(T.tensor3, pick_z(), g['out'])

    score_true = ll.get_output(d['out'], inputs=x)
    score_fake = ll.get_output(d['out'], inputs=x_fake)

    print("=> Now testing score_true", end=" ")
    f_test(x, pick_x(), score_true)
    print("=> Now testing score_fake", end=" ")
    f_test(x_fake, pick_x(), score_fake)

    t = srng.uniform((batch_size, )).dimshuffle(0, 'x')

    x_btwn = t * x + (1 - t) * x_fake
    score_btwn = ll.get_output(d['out'], inputs=x_btwn)

    grad_btwn = theano.grad(score_btwn.sum(), wrt=x_btwn)
    grad_l2 = T.sqrt((grad_btwn ** 2).sum(axis=1))

    gamma = 0.1

    obj_g = -T.mean(score_fake)
    print("=> Now testing obj_g", end=" ")
    f_test(x_fake, pick_x(), obj_g)

    obj_d = -T.mean(score_true - score_fake - gamma * (grad_l2 - 1) ** 2)
    print("=> Now testing obj_d", end=" ")
    f_test([x, x_fake], [pick_x(), pick_x()], obj_d)

    params_g = ll.get_all_params(g['out'], trainable=True)
    params_d = ll.get_all_params(d['out'], trainable=True)

    # variables symboliques, permet d'ajuster le learning rate pendant
    # l'apprentissage si besoin
    lr_g = T.scalar(dtype=config.floatX)
    lr_d = T.scalar(dtype=config.floatX)

    updates_g = lasagne.updates.rmsprop(obj_g, params_g, lr_g)
    updates_d = lasagne.updates.rmsprop(obj_d, params_d, lr_d)


    ## Inverse
    z_rev = T.tensor(dtype=config.floatX, broadcastable=(False, False, False))
    x_rev = T.tensor(dtype=config.floatX, broadcastable=(False, False, False))

    x_fake_rev = ll.get_output(g_rev['out'], inputs=z_rev)
    print("=> Now testing x_fake", end=" ")
    f_test(T.tensor3, pick_z_rev(), g_rev['out'])

    score_true_rev = ll.get_output(d_rev['out'], inputs=x_rev)
    score_fake_rev = ll.get_output(d_rev['out'], inputs=x_fake_rev)

    print("=> Now testing score_true", end=" ")
    f_test(x_rev, pick_x_rev(), score_true_rev)
    print("=> Now testing score_fake", end=" ")
    f_test(x_fake_rev, pick_x_rev(), score_fake_rev)

    t_rev = srng.uniform((batch_size, )).dimshuffle(0, 'x')

    x_btwn_rev = t_rev * x_rev + (1 - t_rev) * x_fake_rev
    score_btwn_rev = ll.get_output(d_rev['out'], inputs=x_btwn_rev)

    grad_btwn_rev = theano.grad(score_btwn_rev.sum(), wrt=x_btwn_rev)
    grad_l2_rev = T.sqrt((grad_btwn_rev ** 2).sum(axis=1))

    gamma = 0.1

    obj_g_rev = -T.mean(score_fake_rev)
    print("=> Now testing obj_g", end=" ")
    f_test(x_fake_rev, pick_x_rev(), obj_g_rev)

    obj_d_rev = -T.mean(score_true_rev - score_fake_rev - gamma * (grad_l2_rev - 1) ** 2)
    print("=> Now testing obj_d", end=" ")
    f_test([x_rev, x_fake_rev], [pick_x_rev(), pick_x_rev()], obj_d_rev)

    params_g_rev = ll.get_all_params(g_rev['out'], trainable=True)
    params_d_rev = ll.get_all_params(d_rev['out'], trainable=True)

    # variables symboliques, permet d'ajuster le learning rate pendant
    # l'apprentissage si besoin
    lr_g_rev = T.scalar(dtype=config.floatX)
    lr_d_rev = T.scalar(dtype=config.floatX)

    updates_g_rev = lasagne.updates.rmsprop(obj_g_rev, params_g_rev, lr_g_rev)
    updates_d_rev = lasagne.updates.rmsprop(obj_d_rev, params_d_rev, lr_d_rev)

    print('...compiling for direct')
    if __debug__:
        mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False)
    else:
        mode = None
    train_g = theano.function(
        inputs=[z, lr_g],
        outputs=[obj_g],
        updates=updates_g,
        mode=mode,
    )

    train_d = theano.function(
        inputs=[z, x, lr_d],
        outputs=[obj_d],
        updates=updates_d,
        mode=mode,
    )

    predict_x = theano.function(
        inputs=[z],
        outputs=[ll.get_output(g['out'], inputs=z, deterministic=True)]
    )

    score_x = theano.function(
        inputs=[x],
        outputs=[ll.get_output(d['out'], inputs=x, deterministic=True)]
    )

    if __debug__:
        mode = NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False)
    else:
        mode = None
    train_g_rev = theano.function(
        inputs=[z_rev, lr_g_rev],
        outputs=[obj_g_rev],
        updates=updates_g_rev,
        mode=mode,
    )

    print('...compiling for reverse')
    train_d_rev = theano.function(
        inputs=[z_rev, x_rev, lr_d_rev],
        outputs=[obj_d_rev],
        updates=updates_d_rev,
        mode=mode,
    )

    predict_x_rev = theano.function(
        inputs=[z_rev],
        outputs=[ll.get_output(g_rev['out'], inputs=z_rev, deterministic=True)]
    )

    score_x_rev = theano.function(
        inputs=[x_rev],
        outputs=[ll.get_output(d_rev['out'], inputs=x_rev, deterministic=True)]
    )

    print('...compiling for cycle')
    lr_cycle = T.scalar(dtype=config.floatX)
    f_circ_g = ll.get_output(
            g_rev['out'],
            inputs=ll.get_output(g['out'], inputs=x, deterministic=True),
            )
    g_circ_f = ll.get_output(
            g['out'],
            inputs=ll.get_output(g_rev['out'], inputs=x_rev, deterministic=True),
            )
    cycle_loss = T.mean(abs(f_circ_g - x)) + T.mean(abs(g_circ_f - x_rev))
    updates_cycle = lasagne.updates.rmsprop(cycle_loss, params_g + params_g_rev, lr_cycle)
    train_cycle = theano.function(
            inputs=[x, x_rev, lr_cycle],
            outputs=cycle_loss,
            updates=updates_cycle,
            mode=mode,
            )


    print('...training')
    n_epochs = 30

    lr_d = 5e-5
    lr_g = 5e-5
    lr_c = 5e-5

    # Boucle d'apprentissage
    d_objs = np.zeros(n_epochs)
    g_objs = np.zeros(n_epochs)
    cycle_objs = np.zeros(n_epochs)

    for i in range(n_epochs):

        # train discriminator
        x = pick_x(batch_size)
        z = pick_z(batch_size)
        d_objs[i] = np.mean(train_d(z, x, lr_d))

        # train generator
        z = pick_z(batch_size)
        g_objs[i] = np.mean(train_g(z, lr_g))

        # train cycle
        z = pick_z(batch_size)
        z_rev = pick_z(batch_size)
        cycle_objs[i] = np.mean(train_cycle(z, z_rev, lr_c))

        print("epoch {} / {} : d {} ; g {} ; c {}...".
                format(i + 1, n_epochs, d_objs[i], g_objs[i], cycle_objs[i]))

    plt.figure()
    plt.plot(d_objs, label="d_objs")
    plt.plot(g_objs, label="g_objs")
    plt.legend()
    plt.savefig('cycle_training.png')
    plt.show()

    make = lambda x: denorm(generate(x))
    denoise = lambda x: denorm(decode(encode(norm(x))))
    seeds = [pick_z() for _ in range(1)]
    moves = []
    moves += list(map(make , seeds))
    moves += list(map(denoise, copy(moves)))
    opinions = list(map(lambda x:discriminate(x)[0], moves))
    moves += [denorm(s) for s in seeds]
    print("Opinion :", opinions)
    animation_plot(copy(moves), interval=15.15)

    make = lambda x: denorm(generate_rev(x))
    denoise = lambda x: denorm(decode(encode(norm(x))))
    seeds = [pick_z_rev() for _ in range(1)]
    moves = []
    moves += list(map(make , seeds))
    moves += list(map(denoise, copy(moves)))
    opinions = list(map(lambda x:discriminate_rev(x)[0], moves))
    moves += [denorm(s) for s in seeds]
    print("Opinion_rev :", opinions)
    animation_plot(copy(moves), interval=15.15)
    raise Exception("Stop")
