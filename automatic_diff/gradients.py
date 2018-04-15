import numpy as np

from automatic_diff.dual_number import DualNumber


def partial_der(x, func, idx):
    return func(*[DualNumber(component, int(i == idx)) for i, component in enumerate(x)])


def gradient(x, func):
    x = np.array(x)
    grad = []
    for i in range(len(x)):
        y = partial_der(x, func, i)
        grad.append(y.dx)
    return y.x, grad


def directional_der(dual_number_array, func):
    return func(*[DualNumber(x, d) for x, d in zip(dual_number_array.x, dual_number_array.dx)])
