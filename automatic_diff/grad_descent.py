import numpy as np
from automatic_diff.dual_number import DualNumber
from automatic_diff.gradients import gradient
from automatic_diff.learning_rates import LearningRate


class GradientDescent:

    def __init__(self, func):
        self.func = func
        self.dual_x = None
        self.y = None
        self.dy = None
        self.lr = None
        self.num_iter = 0

    @property
    def dual_x(self):
        return self.__dual_x

    @property
    def x(self):
        return None if self.dual_x is None else self.dual_x.x

    @property
    def dx(self):
        return None if self.dual_x is None else self.dual_x.dx

    @dual_x.setter
    def dual_x(self, dual_x):
        self.__dual_x = dual_x

    @property
    def lr(self):
        return self.__lr

    @lr.setter
    def lr(self, lr):
        self.__lr = LearningRate.create(lr)
        self.__lr.num_iters = 0

    def fit(self, initial_x=None, tol=1e-2, max_iters=10, lr=0.1, verbose=False):
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


def grad_descent(initial_x, func, **kwargs):
    GD = GradientDescent(func)
    GD.fit(initial_x, **kwargs)
    return GD.dual_x, GD.y, GD.dy