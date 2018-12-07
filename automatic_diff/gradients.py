'''
Gradients and partial derivatives for Dual Numbers
'''
import numpy as np
from automatic_diff.dual_number import DualNumber


def partial_der(x, func, idx):
    '''
    Parameters
    ----------
    x: float or np.array
        Input variable
    func: function
        Dual-Number implemented function that takes input in shape of `x`
    idx: int
        Index of variable to take partial derivative with respect to

    Returns
    -------
    DualNumber
        (x, dx) = (value of `func(x)`, partial derivative of `func` wrt x[`idx`]
    '''
    return func(*[DualNumber(component, int(i == idx)) for i, component in enumerate(x)])


def gradient(x, func):
    '''
    Parameters
    ----------
    x: float or np.array
        Input variable
    func: function
        Dual-Number implemented function that takes input in shape of `x`

    Returns
    -------
    tuple: (float, array of floats)
        First element is the evaluation `func(x)`
        Second element is the list of partial derivatives of `func`
    '''
    x = np.array(x)
    grad = []
    for i in range(len(x)):
        y = partial_der(x, func, i)
        grad.append(y.dx)
    return y.x, grad


def directional_der(dual_number_array, func):
    '''
    Parameters
    ----------
    dual_number_array:
        DualNumber, (x, dx), where dx is the direction vector for the directional derivative
    func: function
        Dual-Number implemented function that takes input in shape of `dual_number_array.x`

    Returns
    -------
    DualNumber
        (x, dx) = (eval of func(x), directional derivative)
    '''
    return func(*[DualNumber(x, d) for x, d in zip(dual_number_array.x, dual_number_array.dx)])
