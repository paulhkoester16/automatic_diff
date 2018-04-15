import numpy as np
from automatic_diff.dual_number import DualNumber


def identity(d: DualNumber):
    return d


def softsign(d: DualNumber):
    return d/(1 + abs(d))


def isru(d: DualNumber, alpha=1.0):
    return d/(1 + alpha * d**2)**0.5

