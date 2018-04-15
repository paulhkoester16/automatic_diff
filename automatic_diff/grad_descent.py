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
        num_iter = 0
        self.lr = lr
        dual_x = DualNumber.create(initial_x)
        dual_x.dx = 2 * tol * np.ones_like(dual_x.dx)
        while num_iter < max_iters and dual_x.size_dx > tol:
            num_iter += 1
            dual_x = self._grad_descent_step(dual_x.x)
            if verbose:
                print("Iteration {}\n\tx:  {}\n".format(num_iter, dual_x))
        if verbose:
            print("y:  {}\n".format(self.y))
        self.dual_x = dual_x

    def _grad_descent_step(self, x):
        self.y, self.dy = gradient(x, self.func)
        new_dx = - np.array(self.dy) * self.lr.lr
        new_x = x + new_dx
        self.lr.update(x=new_x, dx=new_dx, y=self.y, dy=self.dy)
        return DualNumber(new_x, new_dx)


def grad_descent(initial_x, func, **kwargs):
    GD = GradientDescent(func)
    GD.fit(initial_x, **kwargs)
    return GD.dual_x, GD.y, GD.dy