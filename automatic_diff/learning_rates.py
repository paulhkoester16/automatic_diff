'''
Classes for various learning rate schedules
'''
import numpy as np


class LearningRate:
    '''
    Base class for learning rates

    Base class implements constant learning rate for standard gradient descent.
    Child classes can be implemented to add complexity to the learning rate
    schedule, for example decaying learning rates or momentum

    Parameters
    ----------
    lr: float
        Learning rate for standard gradient descent algorithms
    '''
    @classmethod
    def create(cls, *arg):
        '''
        Learning rate factory

        `arg` is expected to either be the arg list for a LearningRate
        or an already instantiated LearningRate
        '''
        if isinstance(arg[0], cls):
            val = arg[0]
        else:
            val = cls(*arg)
        return val

    def __init__(self, lr=0.1):
        self._init_lr = lr
        self.lr = self._init_lr

    @property
    def lr(self): # pylint: disable=missing-docstring
        return self.__lr

    @lr.setter
    def lr(self, lr):
        self.__lr = lr # pylint: disable=attribute-defined-outside-init

    def update(self, grad_descent):
        '''
        Update for standard gradient descent

        Gradient step is learning rate * gradient step of f(x)

        Parameters
        ----------
        grad_descent: gradient_descent.GradientDescent

        Returns
        -------
        np.array
            Step vector for next iteration
        '''
        return np.array(grad_descent.dy) * self.lr


class TimeDecayLearningRate(LearningRate):
    '''
    Similar to LearningRate, except the learning rate decays with the number of iterations

    Parameters
    ----------
    decay_rate: float
        One iteration `n`, learning rate decays by a further factor of
        (1 + `decay_rate` * `n`)
    kwargs:
        Keyword arguments passed to `LearningRate`'s constructor
    '''

    def __init__(self, decay_rate=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.decay_rate = decay_rate

    def update(self, grad_descent):
        '''
        Parameters
        ----------
        grad_descent: gradient_descent.GradientDescent

        Returns
        -------
        np.array
            Step vector for next iteration
        '''
        self.lr = self._init_lr / (1 + self.decay_rate * grad_descent.num_iter)
        return np.array(grad_descent.dy) * self.lr


class GradDecayLearningRate(LearningRate):
    '''
    Similar to LearningRate, except the learning rate decays based recent
    gradient steps.

    Need to recall references for this particular version.

    Parameters
    ----------
    patience: int
        Learning rate is based on the `patience` most recent gradient steps
    '''
    def __init__(self, patience=5, **kwargs):
        super().__init__(**kwargs)
        self.patience = patience
        self.this_dy = None
        self.prev_dy = None
        self.this_x = None
        self.prev_x = None
        self.prev_lr = []

    def update(self, grad_descent):
        '''
        Parameters
        ----------
        grad_descent: gradient_descent.GradientDescent

        Returns
        -------
        np.array
            Step vector for next iteration
        '''
        if grad_descent.num_iter > 0:
            self.prev_dy = self.this_dy.copy()
            self.prev_x = self.this_x.copy()
        self.this_dy = np.array(grad_descent.dy)
        self.this_x = np.array(grad_descent.x)
        if grad_descent.num_iter > 0:
            delta_dy = self.this_dy - self.prev_dy
            delta_x = self.this_x - self.prev_x
            self.lr = np.dot(delta_x, delta_dy) / np.dot(delta_dy, delta_dy)
            self.lr = min(abs(self.lr), abs(self._init_lr))
            self.prev_lr.append(self.lr)
            if len(self.prev_lr) > self.patience:
                self.prev_lr.pop(0)
            self.lr = np.mean(self.prev_lr)
        return np.array(grad_descent.dy) * self.lr


class MomentumLearningRate(LearningRate):
    '''
    Similar to LearningRate, except learning rate adapts based on momentum

    Parameters
    ----------
    decay_rate: float
        One iteration `n`, learning rate decays by a further factor of
        (1 + `decay_rate` * `n`)
    momentum_rate: float
        Scale factor for how much previous gradient steps affect next step
    kwargs:
        Keyword arguments passed to `LearningRate`'s constructor
    '''
    def __init__(self, momentum_rate=0.9, decay_rate=1e-1, **kwargs):
        super().__init__(**kwargs)
        self.momentum_rate = momentum_rate
        self.decay_rate = decay_rate
        self.nu = None

    def update(self, grad_descent):
        '''
        Parameters
        ----------
        grad_descent: gradient_descent.GradientDescent

        Returns
        -------
        np.array
            Step vector for next iteration
        '''
        self.lr = self._init_lr / (1 + self.decay_rate * grad_descent.num_iter)
        if grad_descent.num_iter == 0:
            self.nu = np.zeros_like(grad_descent.x)
        self.nu = self.momentum_rate * self.nu + np.array(grad_descent.dy) * self.lr
        # print(self.nu)
        return self.nu
