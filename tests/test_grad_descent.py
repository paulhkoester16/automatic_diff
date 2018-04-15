import unittest
import numpy as np

from automatic_diff.dual_number import DualNumber
from automatic_diff.grad_descent import grad_descent

from tests.utils import DualNumberTestCase


class TestGradDescent(DualNumberTestCase):

    def test_univar_grad_descent(self):
        tol = 1e-4
        func = lambda d: (d - 3)**2 + 5
        initial_x = np.array([10])
        x, y, dy = grad_descent(initial_x, func, max_iters=100, tol=tol, lr=0.6)
        self.assertAlmostEqual(x.x[0], 3, places=1)
        self.assertAlmostEqual(y, 5, places=1)
        self.assertAlmostEqual(x.size_dx, tol, places=1)

    def test_bivar_grad_descent(self):
        tol = 1e-2
        func = lambda d_0, d_1: (d_0 - 2)**2 + (d_1 + 3)**2 + 8
        initial_x = np.array([10, 12])
        x, y, dy = grad_descent(initial_x, func, max_iters=100, tol=tol, lr=0.6)
        self.assertAlmostEqual(x.x[0], 2, places=1)
        self.assertAlmostEqual(x.x[1], -3, places=1)
        self.assertAlmostEqual(y, 8, places=1)
        self.assertAlmostEqual(x.size_dx, tol, places=1)