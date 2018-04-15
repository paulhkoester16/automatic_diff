import unittest
import numpy as np
import numpy.testing as npt

from automatic_diff.dual_number import DualNumber, DualNumberError


class TestDualNumberConstructor(unittest.TestCase):

    def test_constructor_with_numbers(self):
        x = 1
        dx = -2
        dual_number = DualNumber(x, dx)
        self.assertEqual(np.array([x]), dual_number.x)
        self.assertEqual(np.array([dx]), dual_number.dx)

    def test_constructor_with_matrices(self):
        x = np.mat([1, 4])
        dx = np.mat([-2, 3])
        dual_number = DualNumber(x, dx)
        npt.assert_equal(x, dual_number.x)
        npt.assert_equal(dx, dual_number.dx)

    def test_constructor_with_arrays(self):
        x = np.array([[1, 4]])
        dx = np.array([[-2, 3]])
        dual_number = DualNumber(x, dx)
        npt.assert_equal(x, dual_number.x)
        npt.assert_equal(dx, dual_number.dx)

    def test_inconsistent_shape_raise_dual_number_error(self):
        x = np.array([1, 0])
        dx = np.array([[-2], [0]])
        with self.assertRaises(DualNumberError):
            DualNumber(x, dx)

    def test_factory_create_from_number(self):
        x = 3
        dual_number = DualNumber.create(x)
        npt.assert_equal(np.array([x]), dual_number.x)
        npt.assert_equal(np.array([0]), dual_number.dx)

    def test_factory_create_from_array(self):
        x = np.array([[1, 2], [2, 3], [3, 4]])
        dual_number = DualNumber.create(x)
        npt.assert_equal(x, dual_number.x)
        npt.assert_equal(np.zeros_like(x), dual_number.dx)

    def test_factory_create_from_dual_number(self):
        first = DualNumber(3, 2)
        second = DualNumber.create(first)
        self.assertEqual(first, second)


class TestDualNumberUnaryOps(unittest.TestCase):

    def setUp(self):
        self.x = np.array([4, -2, 3, 2])
        self.dx = np.array([-6, 8, 3, 0])
        self.dual_number = DualNumber(self.x, self.dx)

    def test_negation(self):
        actual = - self.dual_number
        expected = DualNumber(-self.x, -self.dx)
        self.assertEqual(expected, actual)

    def test_abs(self):
        actual = abs(self.dual_number)
        expected = DualNumber([4, 2, 3, 2], [-6, -8, 3, 0])
        self.assertEqual(expected, actual)

    @unittest.skip
    def test_round(self):
        actual = round(DualNumber([2.8, 3.13, -1.97, -3.42], [2.8, 3.13, -1.97, -3.42]), 1)
        expected = DualNumber([2.8, 3.1, -2.0, -3.4], [2.8, 3.1, -2.0, -3.4])
        self.assertEqual(expected, actual)

class TestPowerRule(unittest.TestCase):

    def test_power_rule(self):
        x = 4
        dx = 5
        power = 2.5
        dual_number = DualNumber(x, dx)
        actual = dual_number**power
        expected = DualNumber(x**power, dx * power * x**(power - 1))
        self.assertEqual(expected, actual)


class TestLinearityDualNumbers(unittest.TestCase):

    def test_dual_plus_dual(self):
        actual = DualNumber(3, 8) + DualNumber(-2, 4)
        expected = DualNumber(3 - 2, 8 + 4)
        self.assertEqual(expected, actual)

    def test_dual_plus_number(self):
        actual = DualNumber(3, 8) + 2
        expected = DualNumber(3 + 2, 8 + 0)
        self.assertEqual(expected, actual)

    def test_number_plus_dual(self):
        actual = 2 + DualNumber(3, 8)
        expected = DualNumber(3 + 2, 8 + 0)
        self.assertEqual(expected, actual)

    def test_dual_minus_dual(self):
        actual = DualNumber(3, 8) - DualNumber(-2, 4)
        expected = DualNumber(3 - -2, 8 - 4)
        self.assertEqual(expected, actual)

    def test_dual_minus_number(self):
        actual = DualNumber(3, 8) - 2
        expected = DualNumber(3 - 2, 8 - 0)
        self.assertEqual(expected, actual)

    def test_number_minus_dual(self):
        actual = 2 - DualNumber(3, 8)
        expected = DualNumber(2 - 3, 0 - 8)
        self.assertEqual(expected, actual)

    def test_dual_times_scalar(self):
        actual = DualNumber(3, 8) * -5
        expected = DualNumber(-5*3, -5*8)
        self.assertEqual(expected, actual)

    def test_scalar_times_dual(self):
        actual = -5 * DualNumber(3, 8)
        expected = DualNumber(-5*3, -5*8)
        self.assertEqual(expected, actual)

    def test_dual_divide_by_scalar(self):
        actual = DualNumber(3, 8) / -5
        expected = DualNumber(3 / -5, 8 / -5)
        self.assertEqual(expected, actual)


class TestProductRule(unittest.TestCase):

    def test_dual_times_dual(self):
        x1 = 4
        dx1 = -5
        x2 = 9
        dx2 = 2
        actual = DualNumber(x1, dx1) * DualNumber(x2, dx2)
        expected = DualNumber(x1 * x2, x1 * dx2 + x2 * dx1)
        self.assertEqual(expected, actual)


class TestQuotientRule(unittest.TestCase):

    def test_dual_divide_by_dual(self):
        x1 = 4
        dx1 = -5
        x2 = 9
        dx2 = 2
        actual = DualNumber(x1, dx1) / DualNumber(x2, dx2)
        expected = DualNumber(x1 / x2, (x2 * dx1 -  x1 * dx2) / x2**2)
        self.assertEqual(expected, actual)

    def test_scalar_divide_by_dual(self):
        x1 = 4
        x2 = 9
        dx2 = 2
        actual = x1 / DualNumber(x2, dx2)
        expected = DualNumber(x1 / x2, -x1 * dx2/ x2**2)
        self.assertEqual(expected, actual)