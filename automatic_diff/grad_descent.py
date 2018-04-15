import numpy as np
from automatic_diff.dual_number import DualNumber
from automatic_diff.gradients import gradient


def _grad_descent_step(x, func, lr=0.01):
    y, grad = gradient(x, func)
    new_dx = - np.array(grad) * lr
    new_x = x + new_dx
    return DualNumber(new_x, new_dx)


def grad_descent(initial_x, func, tol=1e-2, max_iters=10, lr=0.1):
    num_iter = 0
    dual_x = DualNumber.create(initial_x)
    dual_x.dx = 2 * tol * np.ones_like(dual_x.dx)
    while num_iter < max_iters and dual_x.size_dx > tol:
        num_iter += 1
        dual_x = _grad_descent_step(dual_x.x, func, lr)
        print("Iteration {}".format(num_iter))
        print("\tx:  {}".format(dual_x))
    y, dy = gradient(dual_x.x, func)

    print("y:  {}".format(y))
    return dual_x, y, dy