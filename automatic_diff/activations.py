'''
Standard library of activation functions, implemented for dual numbers.

https://en.wikipedia.org/wiki/Activation_function
'''
from automatic_diff.dual_number import DualNumber


def identity(d: DualNumber):
    '''Identity activation'''
    return d


def softsign(d: DualNumber):
    '''Softsign activation'''
    return d/(1 + abs(d))


def isru(d: DualNumber, alpha=1.0):
    '''Inverse square root unit'''
    return d/(1 + alpha * d**2)**0.5
