import numpy as np

from automatic_diff import gradients as grads
from automatic_diff import functions as fn
from automatic_diff.dual_number import DualNumber
from tests.utils import DualNumberTestCase


class TestGradientBivariateFunction(DualNumberTestCase):

    def setUp(self):
        self.func = lambda d_0, d_1: d_0 * d_1 + fn.sin(d_0)
        self.x = [np.pi/3, 7]
        self.y = self.x[0] * self.x[1] + np.sin(self.x[0])

    def test_x_partial_derivative(self):
        actual = grads.partial_der(self.x, self.func, 0)
        expected = DualNumber(self.y, self.x[1] + np.cos(self.x[0]))
        self.assertAlmostEqual(expected, actual)

    def test_y_partial_derivative(self):
        actual = grads.partial_der(self.x, self.func, 1)
        expected = DualNumber(self.y, self.x[0])
        self.assertAlmostEqual(expected, actual)

    def test_gradient(self):
        y, grad = grads.gradient(self.x, self.func)
        self.assertEqual(self.y, y)
        self.assertEqual([self.x[1] + np.cos(self.x[0]), self.x[0]], grad)

    def test_directional_derivative(self):
        y, grad = grads.gradient(self.x, self.func)
        direction = [3/5, 4/5]
        actual = np.dot(np.array(grad), direction)
        expected = grads.directional_der(DualNumber(self.x, direction), self.func).dx
        self.assertEqual(expected, actual)