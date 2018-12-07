'''
Classes for LinearRegression
'''
import abc
import numpy as np
from automatic_diff.grad_descent import grad_descent


class Model(metaclass=abc.ABCMeta):
    '''
    Parameters
    ----------
    X: array-like
        number of records by number of features
    y: array-like
        labels corresponding to X
    init_params:
        Initial values for the model's parameters that will be learned
    '''
    def __init__(self, X, y, init_params=None):
        self.X = X
        self.y = y
        self.init_params = init_params

    @property
    def init_params(self): # pylint: disable=missing-docstring
        return self.__init_params

    @init_params.setter
    def init_params(self, init_params):
        self.__init_params = init_params # pylint: disable=attribute-defined-outside-init

    def fit(self, *args, **kwargs):
        '''
        Parameters
        ----------
        args, kwargs:
            Passed to `grad_descent`

        Returns
        -------
        Learned values of model's parameters
        '''
        params, loss, dloss = grad_descent(self.init_params, self.loss_func, *args, **kwargs) # pylint: disable=unused-variable
        return params

    @abc.abstractmethod
    def loss_func(self, *params):
        '''loss function by which model is optimized against'''
        pass


class LinearRegression(Model):
    '''
    Least squares linear regression
    '''
    @property
    def init_params(self): # pylint: disable=missing-docstring
        return self.__init_params

    @init_params.setter
    def init_params(self, init_params=None):
        if not init_params:
            init_params = np.random.uniform(-1, 1, len(self.X) + 1)
        self.__init_params = init_params # pylint: disable=attribute-defined-outside-init

    def loss_func(self, *params):
        return sum(
            (sum(s * component for s, component in zip(params[1:], x)) + params[0] - y)**2
            for x, y in zip(self.X, self.y)
        )**0.5
