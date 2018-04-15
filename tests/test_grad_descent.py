import unittest
import numpy as np

from automatic_diff.dual_number import DualNumber
import automatic_diff.activations as act
from automatic_diff.grad_descent import grad_descent

from tests.utils import DualNumberTestCase


class TestGradDescent(DualNumberTestCase):

    def test_univar_grad_descent(self):
        tol = 1e-4
        func = lambda d: (d - 3)**2 + 5
        initial_x = np.array([10])
        x, y = grad_descent(initial_x, func, max_iters=100, tol=tol, lr=0.6)
        self.assertAlmostEqual(abs(x.x)[0], 3, places=1)
        self.assertAlmostEqual(abs(y.x)[0], 5, places=1)
        self.assertAlmostEqual(abs(x.dx)[0], tol, places=1)

    @unittest.skip
    def test_bivar_grad_descent(self):
        tol = 1e-2
        func = lambda d: (d - 3)**2 + 5
        initial_x = np.array([10, 5])
        x, y = grad_descent(initial_x, func, max_iters=100, tol=tol, lr=0.6)
        self.assertAlmostEqual(abs(x.x), 3, places=1)
        self.assertAlmostEqual(abs(y.x), 5, places=1)
        self.assertAlmostEqual(abs(x.dx), tol, places=1)