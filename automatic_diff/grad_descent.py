import numpy as np
from automatic_diff.dual_number import DualNumber


def _grad_descent_step(x, func, lr=0.01):
    y = func(x)
    new_dx = - y.dx * lr
    new_x = x.x + new_dx
    return DualNumber(new_x, new_dx)


def grad_descent(initial_x, func, tol=1e-2, max_iters=10, lr=0.1):
    num_iter = 0
    x = DualNumber(initial_x, np.array([1]))
    while abs(x.dx) > tol and num_iter < max_iters:
        num_iter += 1
        x.dx = np.array([1])
        x = _grad_descent_step(x, func, lr)
        print("Iteration {}".format(num_iter))
        print("\tx:  {}".format(x))
    y = func(x)
    print("y:  {}".format(y))
    return x, y