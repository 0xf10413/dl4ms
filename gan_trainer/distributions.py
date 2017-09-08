#!/usr/bin/env python3
import os

import numpy as np
import scipy
from sklearn import mixture

os.environ['THEANO_FLAGS'] = (
    #'device=cuda'
    'device=cpu'
)

from theano import config

class Distribution(object):
    """
    Classe abstraite représentant une distribution de probas
    Membres :
    -> rng, générateur de distribution aléatoire
    -> _pdf, vraie pdf de la distribution (peut ne pas être connu)
    """
    def __init__(self, rng, sample_size):
        self.rng = rng
        self.sample_size = sample_size
        self._pdf = None

    def sample(self, sample_size):
        raise NotImplementedError("Did not extend base method")

    def pdf(self, interval):
        if self._pdf is None:
            raise NotImplementedError("No known pdf for this distribution")
        return self._pdf(interval)

    def __add__(self, other):
        if self._pdf is None or other._pdf is None:
            raise NotImplementedError("Cant add {} and {}".format(self, other))
        if self.rng is not other.rng:
            raise RuntimeError("Inconsistency in rng")
        distr = Distribution(self.rng, min(self.sample_size, other.sample_size))
        distr._pdf = lambda x: np.convolve(self._pdf(x), other._pdf(x), 'same')
        distr.sample = lambda size: self.sample(size) + other.sample(size)
        return distr

    def __pos__(self):
        return self


class GaussianDistribution(Distribution):
    def __init__(self, rng, sample_size, *, mean, scale):
        super().__init__(rng, sample_size)
        self.mean = mean
        self.scale = scale
        self._pdf = lambda x: scipy.stats.norm.pdf(x, loc=self.mean,
                scale=self.scale).astype(config.floatX)

    def sample(self, sample_size):
        return self.rng.normal(loc=self.mean, scale=self.scale,
                size=sample_size).astype(config.floatX)


class UniformDistribution(Distribution):
    def __init__(self, rng, sample_size, *, a, b):
        super().__init__(rng, sample_size)
        assert a != b
        self.min = min(a, b)
        self.max = max(a, b)
        self._pdf = lambda x: np.logical_and(self.min <= x, x <= self.max)/(self.max-self.min)

    def sample(self, sample_size):
        return self.rng.uniform(low=self.min, high=self.max,
                size=sample_size).astype(config.floatX)

class LaplaceDistribution(Distribution):
    def __init__(self, rng, sample_size, *, mean, scale):
        super().__init__(rng, sample_size)
        self.mean = mean
        self.scale = scale
        self._pdf = lambda x: np.exp(-abs(x-self.mean)/self.scale)/(2.*self.scale)

    def sample(self, sample_size):
        return self.rng.laplace(loc=self.mean, scale=self.scale,
                size=sample_size).astype(config.floatX)


class GaussianMixture(Distribution):
    def __init__(self, rng, sample_size, *, means, scales):
        super().__init__(rng, sample_size)
        assert len(means) == len(scales)
        self.means = means
        self.scales = scales
        self.gaussians = [GaussianDistribution(rng, sample_size, mean=m, scale=s)
                for (m,s) in zip (means, scales)]
        self._pdf = lambda x: sum([g.pdf(x) for g in self.gaussians])

    def sample(self, sample_size):
        n = len(self.gaussians)
        if isinstance(sample_size, tuple):
            sample_size = (sample_size[0]//n, *sample_size[1:])
        else:
            sample_size //= n
        output = [self.gaussians[i].sample(sample_size) for i in range(n)]
        return np.concatenate(output, axis=0)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    X = GaussianMixture(np.random.RandomState(4), 10, means=[-.5, .5], scales=[.1, .1])
    plt.hist(X.sample(10000), bins=100)
    plt.show()
