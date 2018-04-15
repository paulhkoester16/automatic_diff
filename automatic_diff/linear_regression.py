import abc
import numpy as np

from automatic_diff.grad_descent import grad_descent


class Model(metaclass=abc.ABCMeta):

    def __init__(self, X, y, init_params=None):
        self.X = X
        self.y = y
        self.init_params = init_params

    @property
    def init_params(self):
        return self.__init_params

    @init_params.setter
    def init_params(self, init_params):
        self.__init_params = init_params

    def fit(self, *args, **kwargs):
        params, loss, dloss = grad_descent(self.init_params, self.loss_func, *args, **kwargs)
        return params

    @abc.abstractmethod
    def loss_func(self, *params):
        pass


class LinearRegression(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def init_params(self):
        return self.__init_params

    @init_params.setter
    def init_params(self, init_params=None):
        if not init_params:
            init_params = np.random.uniform(-1, 1, len(self.X) + 1)
        self.__init_params = init_params

    def loss_func(self, *params):
        return sum(
            (sum(s * component for s, component in zip(params[1:], x)) + params[0] - y)**2
            for x, y in zip(self.X, self.y)
        )**0.5
