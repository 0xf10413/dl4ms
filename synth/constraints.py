#!/usr/bin/env python3
import sys
import time
import numpy as np
import scipy.io as io
import theano
import theano.tensor as T

def multiconstraint(*fs):
    return lambda H, V: (sum(map(lambda f: f(H, V), fs)) / len(fs))

def trajectory(traj):

    def trajectory_constraint(H, V):
        velocity_scale = 10
        return velocity_scale * T.mean((V[:,-7:-4] - traj)**2)

    return trajectory_constraint

def foot_sliding(labels):

    def foot_sliding_constraint(H, V):

        feet = np.array([[12,13,14], [15,16,17],[24,25,26], [27,28,29]])
        contact = (labels > 0.5).astype(theano.config.floatX)

        offsets = T.concatenate([
            V[:,feet[:,0:1]],
            T.basic.zeros((V.shape[0],len(feet),1,V.shape[2])),
            V[:,feet[:,2:3]]], axis=2)

        def cross(A, B):
            return T.concatenate([
                A[:,:,1:2]*B[:,:,2:3] - A[:,:,2:3]*B[:,:,1:2],
                A[:,:,2:3]*B[:,:,0:1] - A[:,:,0:1]*B[:,:,2:3],
                A[:,:,0:1]*B[:,:,1:2] - A[:,:,1:2]*B[:,:,0:1]
            ], axis=2)

        rotation = -V[:,-5].dimshuffle(0,'x','x',1) * cross(np.array([[[0,1,0]]]), offsets)

        velocity_scale = 10
        cost_feet_x = velocity_scale * T.mean(contact[:,:,:-1] * (((V[:,feet[:,0],1:] - V[:,feet[:,0],:-1]) + V[:,-7,:-1].dimshuffle(0,'x',1) + rotation[:,:,0,:-1])**2))
        cost_feet_z = velocity_scale * T.mean(contact[:,:,:-1] * (((V[:,feet[:,2],1:] - V[:,feet[:,2],:-1]) + V[:,-6,:-1].dimshuffle(0,'x',1) + rotation[:,:,2,:-1])**2))
        #cost_feet_y = T.mean(contact * ((V[:,feet[:,1]] - np.array([[0.75], [0.0], [0.75], [0.0]]))**2))
        cost_feet_y = velocity_scale * T.mean(contact[:,:,:-1] * ((V[:,feet[:,1],1:] - V[:,feet[:,1],:-1]) **2))
        cost_feet_h = 10.0 * T.mean(T.minimum(V[:,feet[:,1],1:], 0.0)**2)

        return (cost_feet_x + cost_feet_z + cost_feet_y + cost_feet_h) / 4

    return foot_sliding_constraint

def joint_lengths(
    parents=np.array([-1,0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]),
    lengths=np.array([
            2.40,7.15,7.49,2.36,2.37,7.43,7.50,2.41,
            2.04,2.05,1.75,1.76,2.90,4.98,3.48,0.71,
            2.73,5.24,3.44,0.62], dtype=theano.config.floatX)):

    def joint_lengths_constraint(H, V):

        J = V[:,:-7].reshape((V.shape[0], len(parents), 3, V.shape[2]))
        return T.mean((
            T.sqrt(T.sum((J[:,2:] - J[:,parents[2:]])**2, axis=2)) -
            lengths[np.newaxis,...,np.newaxis])**2)

    return joint_lengths_constraint

## flo: contient un c-c de nn/AdamTrainer
def constrain(X, forward, backward, preprocess, constraint, alpha=0.1, iterations=100):

    H = theano.shared(np.array(forward(theano.shared((X - preprocess['Xmean']) / preprocess['Xstd'])).eval()))
    V = (backward(H) * preprocess['Xstd']) + preprocess['Xmean']

    cost = constraint(H, V)

    self_alpha = alpha
    self_beta1 = 0.9
    self_beta2 = 0.999
    self_eps = 1e-05
    self_batchsize = 1

    self_params = [H]
    self_m0params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX)) for p in self_params]
    self_m1params = [theano.shared(np.zeros(p.shape.eval(), dtype=theano.config.floatX)) for p in self_params]
    self_t = theano.shared(np.array([1], dtype=theano.config.floatX))

    gparams = T.grad(cost, self_params)
    m0params = [self_beta1 * m0p + (1-self_beta1) *  gp     for m0p, gp in zip(self_m0params, gparams)]
    m1params = [self_beta2 * m1p + (1-self_beta2) * (gp*gp) for m1p, gp in zip(self_m1params, gparams)]
    params = [p - (self_alpha / self_batchsize) *
              ((m0p/(1-(self_beta1**self_t[0]))) /
        (T.sqrt(m1p/(1-(self_beta2**self_t[0]))) + self_eps))
        for p, m0p, m1p in zip(self_params, m0params, m1params)]

    updates = ([( p,  pn) for  p,  pn in zip(self_params, params)] +
               [(m0, m0n) for m0, m0n in zip(self_m0params, m0params)] +
               [(m1, m1n) for m1, m1n in zip(self_m1params, m1params)] +
               [(self_t, self_t+1)])

    constraint_func = theano.function([], cost, updates=updates)
    start = time.clock()
    for i in range(iterations):
       cost = constraint_func()
       print('Constraint Iteration %i, error %f' % (i, cost))
    print('Constraint: %0.4f' % (time.clock() - start))

    return (np.array(backward(H).eval()) * preprocess['Xstd']) + preprocess['Xmean']


