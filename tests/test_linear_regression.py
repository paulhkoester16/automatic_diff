import numpy as np
from sklearn import linear_model

from automatic_diff.linear_regression import LinearRegression
import automatic_diff.learning_rates as learn_rates
from tests.utils import DualNumberTestCase


class TestLinearRegression(DualNumberTestCase):

    def setUp(self):
        self.X = [0, 1, 2, 3, 4, 5, 6]
        self.slope = 2.5
        self.intercept = 4
        self.noise = [0.1, -0.1, -0.1, 0.1, -0.1, 0.0, 0.1]
        self.y = [self.slope * x + self.intercept + n for x, n in zip(self.X, self.noise)]

    def test_fit_via_sklearn(self):
        clf = linear_model.LinearRegression()
        X = np.array([[x] for x in self.X])
        y = np.array(self.y)
        clf.fit(X, y)
        slope = clf.coef_
        intercept = clf.intercept_
        self.assertAlmostEqual(self.slope, slope[0], places=1)
        self.assertAlmostEqual(self.intercept, intercept, places=1)

    def test_noise(self):
        self.assertAlmostEqual(sum(self.noise), 0)

    def test_fit_linear_regression_via_class(self):
        model = LinearRegression([[x] for x in self.X], self.y, init_params=[0, 1])
        params = model.fit(max_iters=1000, tol=1e-3, lr=0.005, verbose=False)
        self.assertAlmostEqual(params.x[0], self.intercept, places=1)
        self.assertAlmostEqual(params.x[1], self.slope, places=1)

    def test_fit_linear_regression_time_decaying_lr(self):
        model = LinearRegression([[x] for x in self.X], self.y, init_params=[0, 1])
        lr = learn_rates.TimeDecayLearningRate(lr=0.1, decay_rate=0.1)
        params = model.fit(max_iters=1000, tol=1e-3, lr=lr, verbose=False)
        self.assertAlmostEqual(params.x[0], self.intercept, places=1)
        self.assertAlmostEqual(params.x[1], self.slope, places=1)

    def test_fit_linear_regression_grad_decaying_lr(self):
        model = LinearRegression([[x] for x in self.X], self.y, init_params=[0, 1])
        lr = learn_rates.GradDecayLearningRate(lr=0.1)
        params = model.fit(max_iters=1000, tol=1e-5, lr=lr, verbose=False)
        self.assertAlmostEqual(params.x[0], self.intercept, places=1)
        self.assertAlmostEqual(params.x[1], self.slope, places=1)

    def test_fit_linear_regression_momentum_lr(self):
        model = LinearRegression([[x] for x in self.X], self.y, init_params=[0, 1])
        lr = learn_rates.MomentumLearningRate(lr=0.1, momentum_rate=0.9)
        params = model.fit(max_iters=1000, tol=1e-5, lr=lr, verbose=True)
        self.assertAlmostEqual(params.x[0], self.intercept, places=1)
        self.assertAlmostEqual(params.x[1], self.slope, places=1)

    def test_init_linear_regression_with_params(self):
        model = LinearRegression([[x] for x in self.X], self.y, init_params=[0, 1])
        self.assertEquals([0, 1], model.init_params)

    def test_init_linear_regression_without_params(self):
        model = LinearRegression([[x] for x in self.X], self.y, init_params=None)
        self.assertEquals(2, len(model.init_params))

