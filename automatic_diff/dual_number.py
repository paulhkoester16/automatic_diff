import numpy as np


class DualNumberError(Exception):
    pass


class DualNumber:

    @classmethod
    def create(cls, val):
        if isinstance(val, cls):
            return val
        else:
            x = np.array(val)
            zero = np.zeros_like(x)
            return cls(x=x, dx=zero)

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
    def x(self):
        return self.__x

    @x.setter
    def x(self, value):
        self.__x = np.array(value)

    @property
    def dx(self):
        return self.__dx

    @dx.setter
    def dx(self, value):
        self.__dx = np.array(value)

    @property
    def shape(self):
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