import numpy as np


from automatic_diff.dual_number import DualNumber
import automatic_diff.activations as act
from tests.utils import DualNumberTestCase


class TestActivations(DualNumberTestCase):

    def test_identity(self):
        x, dx = 2, 3
        expected = DualNumber(x, 1 * dx)
        actual = act.identity(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_softsign(self):
        x, dx = 2, 3
        expected = DualNumber(x/(1 + abs(x)), 1/(1 + abs(x))**2 * dx)
        actual = act.softsign(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_isru(self):
        x, dx = 2, 3
        alpha = 4
        expected = DualNumber(x/(1 + alpha * x**2)**0.5, 1/(1 + alpha * x**2)**1.5 * dx)
        actual = act.isru(DualNumber(x, dx), alpha=alpha)
        self.assertAlmostEqual(expected, actual)

    # binary_step
    # sigmoid
    # tanh
    # arctan

