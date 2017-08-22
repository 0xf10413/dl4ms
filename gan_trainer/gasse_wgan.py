"""GAN experiment."""

import os
import numpy as np
import scipy

import matplotlib.pyplot as plt
from matplotlib import animation

os.environ['THEANO_FLAGS'] = (
    #'device=cuda'
    'device=cpu'
)

import theano
import theano.tensor as T
from theano import config
# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
import lasagne.layers as ll
import lasagne.nonlinearities as lnl
import lasagne.init as lin


if __name__ == "__main__":

    rng = np.random.RandomState(43)
    srng = RandomStreams(seed=rng.randint(0, np.iinfo(np.uint32).max))
    lasagne.random.set_rng(rng)

    z_size = 10
    x_size = 1

    print('...building model')

    # Generator
    g = {}

    g['in'] = ll.InputLayer(
        shape=(None, z_size))

    g['h1'] = ll.DenseLayer(
        g['in'],
        num_units=10,
        nonlinearity=lnl.tanh,
        W=lin.GlorotUniform(),
        b=lin.Constant(0),
    )

    g['out'] = ll.DenseLayer(
        g['h1'],
        num_units=x_size,
        nonlinearity=None,
        W=lin.GlorotUniform(),
        b=lin.Constant(0)
    )

    # Discriminator
    d = {}

    d['in'] = ll.InputLayer(
        shape=(None, x_size))

    d['h1'] = ll.DenseLayer(
        d['in'],
        num_units=50,
        nonlinearity=lnl.tanh,
        W=lin.GlorotUniform(),
        b=lin.Constant(0)
    )

    d['h2'] = ll.DenseLayer(
        d['h1'],
        num_units=50,
        nonlinearity=lnl.tanh,
        W=lin.GlorotUniform(),
        b=lin.Constant(0)
    )

    d['out'] = ll.DenseLayer(
        d['h2'],
        num_units=1,
        nonlinearity=None,
        W=lin.GlorotUniform(),
        b=lin.Constant(0)
    )

    print('generator shape:')
    for l in ll.get_all_layers(g['out']):
        print(format(l.output_shape) + ' ' + format(l.__class__))

    print('network shape:')
    for l in ll.get_all_layers(d['out']):
        print(format(l.output_shape) + ' ' + format(l.__class__))

    z = T.tensor(dtype=config.floatX, broadcastable=(False, False))
    x = T.tensor(dtype=config.floatX, broadcastable=(False, False))

    x_fake = ll.get_output(g['out'], inputs=z)

    score_true = ll.get_output(d['out'], inputs=x)
    score_fake = ll.get_output(d['out'], inputs=x_fake)


    batch_size = x.shape[0]
    t = srng.uniform((batch_size, )).dimshuffle(0, 'x')

    x_btwn = t * x + (1 - t) * x_fake
    score_btwn = ll.get_output(d['out'], inputs=x_btwn)

    grad_btwn = theano.grad(score_btwn.sum(), wrt=x_btwn)
    grad_l2 = T.sqrt((grad_btwn ** 2).sum(axis=1))

    gamma = 0.1

    obj_g = -T.mean(score_fake)
    obj_d = -T.mean(score_true - score_fake - gamma * (grad_l2 - 1) ** 2)

    params_g = ll.get_all_params(g['out'], trainable=True)
    params_d = ll.get_all_params(d['out'], trainable=True)

    # variables symboliques, permet d'ajuster le learning rate pendant
    # l'apprentissage si besoin
    lr_g = T.scalar(dtype=config.floatX)
    lr_d = T.scalar(dtype=config.floatX)

    updates_g = lasagne.updates.rmsprop(obj_g, params_g, lr_g)
    updates_d = lasagne.updates.rmsprop(obj_d, params_d, lr_d)

    print('...compiling')
    train_g = theano.function(
        inputs=[z, lr_g],
        outputs=[obj_g],
        updates=updates_g
    )

    train_d = theano.function(
        inputs=[z, x, lr_d],
        outputs=[obj_d],
        updates=updates_d
    )

    predict_x = theano.function(
        inputs=[z],
        outputs=[ll.get_output(g['out'], inputs=z, deterministic=True)]
    )

    score_x = theano.function(
        inputs=[x],
        outputs=[ll.get_output(d['out'], inputs=x, deterministic=True)]
    )

    print('...training')

    batchsize = 50
    n_epochs = 500

    lr_d = 1e-3
    lr_g = 1e-3

    d_steps = 10
    g_steps = 1

    z_min = 0
    z_max = 1

    x_mean = .5
    x_std = .1

    # figure setup
    fig, sub = plt.subplots(nrows=2, ncols=1, squeeze=False)

    x_range = (-5, 5)
    y_range = (0, 1)
    n_samples = 1000

    x_axis = np.linspace(x_range[0], x_range[1], n_samples)

    true_dist = scipy.stats.norm.pdf(x_axis, loc=x_mean, scale=x_std)
    dist_norm = np.max(true_dist)
    true_dist = true_dist / dist_norm  # normalization

    fake_dist, fake_bins = np.histogram(0, bins=n_samples - 1, range=x_range)
    fake_dist = np.repeat(fake_dist, 2)
    fake_bins = np.repeat(fake_bins, 2)[1:-1]

    curves = {}
    sub[0, 0].set_title("Distributions")
    sub[0, 0].set_xlim(x_range)
    sub[0, 0].set_ylim(y_range)
    sub[0, 0].plot(x_range, (0.5, 0.5), '--')
    sub[0, 0].plot(
        x_axis, true_dist, label='true distribution')
    curves['g_dist'] = sub[0, 0].plot(
        fake_bins, fake_dist, label='G estimation')[0]
    curves['d_score'] = sub[0, 0].plot(
        x_axis, np.repeat(0, n_samples), label='D opinion')[0]
    sub[0, 0].legend()

    sub[1, 0].set_xlabel("Epoch")
    curves['d_obj'] = sub[1, 0].plot(
        range(n_epochs), np.repeat(0, n_epochs), label="Objective D")[0]
    curves['g_obj'] = sub[1, 0].plot(
        range(n_epochs), np.repeat(0, n_epochs), label="Objective G")[0]
    sub[1, 0].legend()

    plt.ion()
    plt.show()
    plt.draw()
    plt.pause(.01)

    writer = animation.FFMpegWriter()
    with writer.saving(fig, 'plot.mp4', dpi=300):

        # Boucle d'apprentissage
        d_objs = np.zeros(n_epochs)
        g_objs = np.zeros(n_epochs)
        for i in range(n_epochs):

            print("epoch {} / {}...".format(i + 1, n_epochs))

            # train discriminator
            x = rng.normal(loc=x_mean, scale=x_std,
                           size=(d_steps, batchsize, x_size)
                           ).astype(config.floatX)
            z = rng.uniform(low=z_min, high=z_max,
                            size=(d_steps, batchsize, z_size)
                            ).astype(config.floatX)
            d_objs[i] = np.mean([train_d(z[j], x[j], lr_d)
                                 for j in range(d_steps)])

            # train generator
            z = rng.uniform(low=z_min, high=z_max,
                            size=(g_steps, batchsize, z_size)
                            ).astype(config.floatX)
            g_objs[i] = np.mean([train_g(z[j], lr_g)
                                 for j in range(g_steps)])

            # plot indicators
            sub[0, 0].set_title('epoch {} / {}'.format(i + 1, n_epochs))

            curves['d_obj'].set_ydata(d_objs)
            curves['g_obj'].set_ydata(g_objs)

            sub[1, 0].relim()
            sub[1, 0].autoscale_view()

            x = x_axis[:, None].astype(config.floatX)
            scores = score_x(x)[0]
            scores = (scores - np.min(scores)) / np.ptp(scores)  # normalization
            curves['d_score'].set_ydata(scores)

            z = rng.uniform(low=z_min, high=z_max,
                            size=(n_samples * 10, z_size)
                            ).astype(config.floatX)
            fake_samples = predict_x(z)[0]
            fake_dist = np.histogram(fake_samples,
                                     bins=n_samples - 1, range=x_range,
                                     density=True)[0]
            fake_dist = np.repeat(fake_dist, 2)
            fake_dist = fake_dist / dist_norm
            curves['g_dist'].set_ydata(fake_dist)

            # write to video
            writer.grab_frame()

            # display live
            plt.draw()
            plt.pause(.01)

    plt.ioff()
    plt.show()
