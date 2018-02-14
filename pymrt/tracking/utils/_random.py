""" This file provides several classes for generate random numbers/vectors
for error estimation/simulation.
"""

import abc
import numpy as np


class RandomGenerator:
    """Abstract class for random number/vector generation
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generate(self):
        r""" Generate a random sample
        """
        raise NotImplementedError()


class UniformGenerator(RandomGenerator):
    """ The class used for generating data according to a uniform
    distribution in a square.
    """
    def __init__(self, low, high, n):
        self.low = np.array(low).reshape((n, 1))
        self.high = np.array(high).reshape((n, 1))
        self.dim = n

    def generate(self):
        return np.random.uniform(self.low, self.high)


class NormalGenerator(RandomGenerator):
    """ NormalGenerator generate random vector from a multi-variate normal
    disribution.
    """
    def __init__(self, mean, cov, n):
        self.mean = np.array(mean).reshape((n, 1))
        self.cov = np.array(cov).reshape((n, n))
        self.n = n

    def generate(self):
        return np.random.multivariate_normal(
            self.mean.reshape((self.n,)), self.cov.T
        ).reshape((self.n, 1))


class PoissonGenerator(RandomGenerator):
    """ Poisson Distribution Generator
    """
    def __init__(self, lam):
        self.lam = lam

    def generate(self):
        return np.random.poisson(lam=self.lam)
