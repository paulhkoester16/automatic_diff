import numpy as np

from automatic_diff.dual_number import DualNumber
import automatic_diff.functions as fn
from tests.utils import DualNumberTestCase


class TestTransCalculus(DualNumberTestCase):

    def test_exp(self):
        x, dx = 2, 3
        expected = DualNumber(np.exp(x), np.exp(x) * dx)
        actual = fn.exp(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_log(self):
        x, dx = 2, 3
        expected = DualNumber(np.log(x), dx / x)
        actual = fn.log(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_sigmoid(self):
        x, dx = 2, 3
        s = 1 / (1 + np.exp(-x))
        expected = DualNumber(s, s*(1 - s) * dx)
        actual = fn.sigmoid(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)


class TestTrigCalculus(DualNumberTestCase):

    def test_sin(self):
        x, dx = 2, 3
        expected = DualNumber(np.sin(x), np.cos(x) * dx)
        actual = fn.sin(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_cos(self):
        x, dx = 2, 3
        expected = DualNumber(np.cos(x), -np.sin(x) * dx)
        actual = fn.cos(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_tan(self):
        x, dx = 2, 3
        expected = DualNumber(np.tan(x), (1/np.cos(x))**2 * dx)
        actual = fn.tan(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_cot(self):
        x, dx = 2, 3
        expected = DualNumber(1/np.tan(x), -(1/np.sin(x))**2 * dx)
        actual = fn.cot(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_csc(self):
        x, dx = 2, 3
        expected = DualNumber(1/np.sin(x), - (1 / np.sin(x)) * (1/ np.tan(x)) * dx)
        actual = fn.csc(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_sec(self):
        x, dx = 2, 3
        expected = DualNumber(1/np.cos(x), (1 / np.cos(x)) * np.tan(x) * dx)
        actual = fn.sec(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)


class TestHyperbolicTrigCalculus(DualNumberTestCase):

    def test_sinh(self):
        x, dx = 2, 3
        expected = DualNumber(np.sinh(x), np.cosh(x) * dx)
        actual = fn.sinh(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_cosh(self):
        x, dx = 2, 3
        expected = DualNumber(np.cosh(x), np.sinh(x) * dx)
        actual = fn.cosh(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_tanh(self):
        x, dx = 2, 3
        expected = DualNumber(np.tanh(x), (1/np.cosh(x))**2 * dx)
        actual = fn.tanh(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_coth(self):
        x, dx = 2, 3
        expected = DualNumber(1/np.tanh(x), -(1/np.sinh(x))**2 * dx)
        actual = fn.coth(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_csch(self):
        x, dx = 2, 3
        expected = DualNumber(1/np.sinh(x), -(1/np.sinh(x)) * (1/np.tanh(x)) * dx)
        actual = fn.csch(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)

    def test_sech(self):
        x, dx = 2, 3
        expected = DualNumber(1/np.cosh(x), -(1/np.cosh(x)) * np.tanh(x) * dx)
        actual = fn.sech(DualNumber(x, dx))
        self.assertAlmostEqual(expected, actual)


class TestChainRule(DualNumberTestCase):

    def test_exp(self):
        x, dx = 2, 3
        inner = fn.sin(DualNumber(x, dx))
        outer = fn.exp(inner)
        actual = outer
        expected = DualNumber(np.exp(np.sin(x)), np.exp(np.sin(x)) * np.cos(x) * dx)
        self.assertAlmostEqual(expected, actual)
