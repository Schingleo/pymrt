import abc
import numpy as np


class ClutterGenerator(object):
    """The clutter class that generates false alarms
    """
    @abc.abstractmethod
    def generate(self):
        raise NotImplementedError()


class CubicUniformPoissonClutter(ClutterGenerator):
    """A clutter process that generates points follows a Poisson Point
    Process, with spacial uniform distribution within a cubic space.
    """
    def __init__(self, lam, low, high, n):
        self.low = low
        self.high = high
        self.n = n
        self.lam = lam

    def generate(self):
        return [
            np.random.uniform(low=self.low, high=self.high, size=(self.n, 1))
            for i in range(np.random.poisson(self.lam))
        ]
