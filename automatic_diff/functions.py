import numpy as np
from automatic_diff.dual_number import DualNumber


def exp(d: DualNumber):
    return DualNumber(np.exp(d.x), np.exp(d.x) * d.dx)


def log(d: DualNumber):
    return DualNumber(np.log(d.x), d.dx / d.x)


def sigmoid(d: DualNumber):
    return 1 / (1 + exp(-d))


def sin(d: DualNumber):
    return DualNumber(np.sin(d.x), np.cos(d.x) * d.dx)


def cos(d: DualNumber):
    return DualNumber(np.cos(d.x), - np.sin(d.x) * d.dx)


def tan(d: DualNumber):
    return sin(d) / cos(d)


def cot(d: DualNumber):
    return 1 / tan(d)


def csc(d: DualNumber):
    return 1 / sin(d)


def sec(d: DualNumber):
    return 1 / cos(d)


def sinh(d: DualNumber):
    return (exp(d) - exp(-d)) / 2


def cosh(d: DualNumber):
    return (exp(d) + exp(-d)) / 2


def tanh(d: DualNumber):
    return sinh(d) / cosh(d)


def coth(d: DualNumber):
    return 1 / tanh(d)


def csch(d: DualNumber):
    return 1 / sinh(d)


def sech(d: DualNumber):
    return 1 / cosh(d)


def matmul(matrix: np.ndarray, d: DualNumber):
    x = d.x if isinstance(d.x, np.ndarray) else np.array([d.x])
    dx = d.dx if isinstance(d.dx, np.ndarray) else np.array([d.dx])
    return DualNumber(np.matmul(matrix, x), np.matmul(matrix, dx))
