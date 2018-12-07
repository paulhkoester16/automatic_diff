'''
Implementation of gradient descent using dual numbers
'''
# pylint: disable=too-many-instance-attributes, too-many-arguments
import numpy as np
from automatic_diff.dual_number import DualNumber
from automatic_diff.gradients import gradient
from automatic_diff.learning_rates import LearningRate


class GradientDescent:
    '''
    Class for gradient descent.

    Casual users would benefit from using the `grad_descent` wrapper function
    rather than this class.

    Parameters
    ----------
    func: function
        Dual-number function to be minimized
    '''
    def __init__(self, func):
        self.func = func
        self.dual_x = None
        self.y = None
        self.dy = None
        self.lr = None
        self.num_iter = 0

    @property
    def dual_x(self): # pylint: disable=missing-docstring
        return self.__dual_x

    @property
    def x(self): # pylint: disable=missing-docstring
        return None if self.dual_x is None else self.dual_x.x

    @property
    def dx(self): # pylint: disable=missing-docstring
        return None if self.dual_x is None else self.dual_x.dx

    @dual_x.setter
    def dual_x(self, dual_x):
        self.__dual_x = dual_x # pylint: disable=attribute-defined-outside-init

    @property
    def lr(self): # pylint: disable=missing-docstring
        return self.__lr

    @lr.setter
    def lr(self, lr):
        self.__lr = LearningRate.create(lr) # pylint: disable=attribute-defined-outside-init
        self.__lr.num_iters = 0

    def fit(self, initial_x=None, tol=1e-2, max_iters=10, lr=0.1, verbose=False):
        '''
        Parameters
        ----------
        initial_x: float or np.array
            Initial value to begin the iterative descent algorithm.
        tol: float
            Tolerance on gradient steps.  If gradient steps are below this, alogorithm
            will terminate.
        max_iters: int
            Maximum number of iterations.  Algorithm may stop early if `tol` is
            reached before `max_iters`.
        lr: float or LearningRate
            If float, just a standard constant learning rate.  Can also specify
            child classes of learning_rates.LearningRate, for finer control over
            adaptive learning rates and momentum.
        verbose: bool
            If true, status messages printed after each iteration.
        '''
        self.num_iter = 0
        self.lr = lr
        self.dual_x = DualNumber.create(initial_x)
        self.dual_x.dx = 2 * tol * np.ones_like(self.dx)
        while self.num_iter < max_iters and self.dual_x.size_dx > tol:
            self._grad_descent_step()
            self.num_iter += 1
            if verbose:
                print("Iteration {}\n\tx:  {}\n".format(self.num_iter, self.dual_x))
        if verbose:
            print("y:  {}\n".format(self.y))

    def _grad_descent_step(self):
        self.y, self.dy = gradient(self.x, self.func)
        new_dx = self.lr.update(self)
        new_x = self.x - new_dx
        self.dual_x = DualNumber(new_x, new_dx)


def grad_descent(initial_x, func, **fit_kwargs):
    '''
    Estimates minimum value of func using gradient descent

    Parameters
    ----------
    initial_x: float or np.array
    func:
        Dual-number function, input same shape as `initial_x`
    fit_kwargs:
        Keyword args passed to `GradientDescent` instance's `fit`

    Returns
    -------
    tuple:
        dual_number: representing the minimizing x and its gradient step
        float: estimated minimum value of function
        float: gradient step of the estimated minimum value
    '''
    grad_desc = GradientDescent(func)
    grad_desc.fit(initial_x, **fit_kwargs)
    return grad_desc.dual_x, grad_desc.y, grad_desc.dy
