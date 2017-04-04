import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams

from Layer import Layer

class Conv1DLayer(Layer):

    def __init__(self, filter_shape, input_shape, stride=(1,), rng=np.random):

        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.stride = stride
        self.theano_rng = RandomStreams(rng.randint(2 ** 30))
        
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])
        
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W = np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX)
        
        self.W = theano.shared(name='W', value=W, borrow=True)
        self.params = [self.W]
    
    def cost(self, input, gamma=0.01):
        return gamma * T.mean(abs(self.W))
    
    def __call__(self, input):
        s, f = self.input_shape, self.filter_shape
        zeros = T.basic.zeros((s[0], s[1], (f[2]-1)//2), dtype=theano.config.floatX)
        input = T.concatenate([zeros, input, zeros], axis=2)
        return conv.conv2d(
            input=input.dimshuffle(0,1,2,'x'),
            filters=self.W.dimshuffle(0,1,2,'x'),
            border_mode='valid',
            subsample=self.stride+(1,))[:,:,:,0]

