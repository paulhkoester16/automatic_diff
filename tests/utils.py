import unittest

from automatic_diff.dual_number import DualNumber


class DualNumberTestCase(unittest.TestCase):

    def assertAlmostEqual(self, first, second, msg=None, **kwargs):
        msg = msg or ''
        if isinstance(first, DualNumber) and isinstance(second, DualNumber):
            msg_x = msg + " comparing dual number's x"
            super().assertAlmostEqual(first.x, second.x, msg=msg_x, **kwargs)
            msg_dx = msg + " comparing dual number's dx"
            super().assertAlmostEqual(first.dx, second.dx, msg=msg_dx, **kwargs)
        else:
            super().assertAlmostEqual(first, second, **kwargs)

