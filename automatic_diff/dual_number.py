'''
Dual Numbers provide (x, dx) tuples with algebraic structure, so that
(x, dx) + (y, dy) = (x + y, dx + dy)
and
(x, dx) * (y, dy) = (x*y, x*dy + y*dx)
'''
import numpy as np


class DualNumberError(Exception):
    '''Error handling for Dual numbers'''
    pass


class DualNumber:
    '''
    Parameters
    ----------
    x: float or np.array
        Corresponds to a variable.
    dx: float or np.array
        Corresponds to gradient step. Must be same shape as `x`
    '''

    @classmethod
    def create(cls, val):
        '''
        Factory for creating DualNumbers

        Parameters
        ----------
        val: Either DualNumber instance or float or np.array
            If float or np.array, initializes a DualNumber with
            `x=val` and `dx=zeros`
        '''
        if not isinstance(val, cls):
            x = np.array(val)
            zero = np.zeros_like(x)
            val = cls(x=x, dx=zero)
        return val

    def __init__(self, x, dx):
        self.x = x
        self.dx = dx
        self._validate()

    def _validate(self):
        if self.x.shape != self.dx.shape:
            raise DualNumberError(
                "x and dx must have same shape but got {} and {}".format(
                    self.x.shape, self.dx.shape))

    @property
    def x(self): # pylint: disable=missing-docstring
        return self.__x

    @x.setter
    def x(self, value):
        self.__x = np.array(value) # pylint: disable=attribute-defined-outside-init

    @property
    def dx(self): # pylint: disable=missing-docstring
        return self.__dx

    @dx.setter
    def dx(self, value):
        self.__dx = np.array(value) # pylint: disable=attribute-defined-outside-init

    @property
    def shape(self): # pylint: disable=missing-docstring
        return self.x.shape

    def __repr__(self):
        return "{} + {} eps".format(self.x, self.dx)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if not np.array_equal(self.x, other.x):
            return False
        if not np.array_equal(self.dx, other.dx):
            return False
        return True

    def __neg__(self):
        return self.__class__(x=-self.x, dx=-self.dx)

    def __abs__(self):
        x = abs(self.x)
        dx = self.dx * np.sign(self.x)
        return self.__class__(x=x, dx=dx)

    def __pow__(self, power):
        x = self.x ** power
        dx = power * self.x ** (power - 1) * self.dx
        return self.__class__(x=x, dx=dx)

    def __add__(self, other):
        other = DualNumber.create(other)
        x = self.x + other.x
        dx = self.dx + other.dx
        return self.__class__(x=x, dx=dx)

    def __radd__(self, other):
        other = DualNumber.create(other)
        return other + self

    def __sub__(self, other):
        other = DualNumber.create(other)
        x = self.x - other.x
        dx = self.dx - other.dx
        return self.__class__(x=x, dx=dx)

    def __rsub__(self, other):
        other = DualNumber.create(other)
        return other - self

    def __mul__(self, other):
        other = DualNumber.create(other)
        x = self.x * other.x
        dx = self.dx * other.x + self.x * other.dx
        return self.__class__(x=x, dx=dx)

    def __rmul__(self, other):
        other = DualNumber.create(other)
        return other * self

    def __truediv__(self, other):
        other = DualNumber.create(other)
        x = self.x / other.x
        dx = (self.dx * other.x - self.x * other.dx) / other.x ** 2
        return self.__class__(x=x, dx=dx)

    def __rtruediv__(self, other):
        other = DualNumber.create(other)
        return other / self

    def __gt__(self, other):
        other = DualNumber.create(other)
        return self.x > other.x

    def __ge__(self, other):
        other = DualNumber.create(other)
        return self.x >= other.x

    def __lt__(self, other):
        other = DualNumber.create(other)
        return self.x < other.x

    def __le__(self, other):
        other = DualNumber.create(other)
        return self.x <= other.x

    @property
    def size_dx(self): # pylint: disable=missing-docstring
        return np.linalg.norm(self.dx)
