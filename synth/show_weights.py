#!/usr/bin/env python3
"""
Montre les coefficients des couches de convolution de l'autoencoder
(couches 1 et 7 en partant de 0)
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../nn')

from Conv1DLayer import Conv1DLayer

from network import create_core

network_core = create_core(batchsize=1, window=240)
network_core.load(np.load('network_core.npz'))

for li, layer in enumerate(network_core.layers[0].layers + network_core.layers[1].layers):

    if not isinstance(layer, Conv1DLayer): continue

    print(li, layer.W.shape.eval())
    print(layer)
    shape = layer.W.shape.eval()
    num = min(shape[0], 32) ## flo: 256, trop de rectangles
    dims = 4, num // 4

    if shape[1] < shape[2]:
        dims = dims[1], dims[0]

    fig, axarr = plt.subplots(dims[0], dims[1], sharex=False, sharey=False)

    W = np.array(layer.W.eval())

    for i in range(dims[0]):
        for j in range(dims[1]):
            axarr[i][j].imshow(
                W[i*dims[1]+j],
                interpolation='nearest', cmap='rainbow',
                vmin=W.mean() - 5*W.std(), vmax=W.mean() + 5*W.std())
            print(W[i*dims[1]+j].shape)
            axarr[i][j].autoscale(False)
            axarr[i][j].grid(False)
            axarr[i][j].legend()
            plt.setp(axarr[i][j].get_xticklabels(), visible=False)
            plt.setp(axarr[i][j].get_yticklabels(), visible=False)

    fig.subplots_adjust(hspace=0)
    fig.tight_layout()
    plt.suptitle('Layer %i Filters' % li, size=16)
    plt.show()

