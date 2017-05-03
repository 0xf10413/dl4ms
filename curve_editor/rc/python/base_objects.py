#!/usr/bin/env python3
"""
Quelques imports et objets utilisés tout le long de l'application
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

# Chemin vers les libs custom
# TODO: vérifier le chemin
sys.path.append('../nn/')
sys.path.append('../synth/')
sys.path.append('../motion/')

# curve, la courbe de déplacement
curve = np.zeros((3, 7200))
curve[0,:] = .5
curve[1,:] = .1
curve[2,:] = np.linspace(0,.2, 7200)

# skel_parents, qui donne la hiérarchie des jointures
skel_parents = np.array(
 [-1,0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20])

# skel, l'ensemble des jointures
skel = np.random.randn(7200, skel_parents.size,3)
skel[:,:,2] = 0
skel[:,:,:2] = [[0,-2],[0,0], # ref, hip
    [-1, -1], [-1, -1.5], [-1, -2], [-1.2, -2], # lfemur, ltibia, lfoot, ltoes
[1, -1], [1, -1.5], [1, -2], [1.2, -2], # rfemur, rtibia, rfoot, rtoes
[0,.2], [0,.4], [0,1], [0, 1.2], # upperback, thorax, upperneck, head
[-1, .5], [-1, .7], [-1.2, .8], [-1.4, .8], # lhumerus, lradius, lwrist, lfingers
[+1, .5], [+1, .7], [+1.2, .8], [+1.4, .8]] # idem à droite

