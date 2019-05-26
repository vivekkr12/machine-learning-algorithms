import unittest

import numpy as np

from pymlalgo.util.model_selection import train_test_split
from pymlalgo.regression.ridge_regression import RidgeRegression
from pymlalgo.util.normalization import Normalizer


class RigdeRegressionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n, cls.d = 1000, 10
        np.random.seed(7)

        x = np.random.uniform(0, 100, cls.n * cls.d).reshape(cls.d, cls.n)
        eps = np.random.normal(loc=0, scale=5, size=cls.n * cls.d).reshape(cls.d, cls.n)
        beta = np.random.uniform(1, 5, cls.d).reshape(cls.d, 1)
        y = (x + eps).T.dot(beta)

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=7, test_size=0.25)

        y_normalizer = Normalizer(x_train, Normalizer.FEATURES)
        cls.x_train_std = y_normalizer.normalize(x_train)
        cls.x_test_std = y_normalizer.normalize(x_test)

        y_normalizer = Normalizer(y_train, Normalizer.LABELS)
        cls.y_train_std = y_normalizer.normalize(y_train)
        cls.y_test_std = y_normalizer.normalize(y_test)

        cls.model = RidgeRegression(cls.x_train_std, cls.y_train_std, lambd=0.1 / cls.n, min_grad=0.00001)
        cls.model.train()

    def test_coefficients(self):
        expected_coef = np.array([0.45926421, 0.28884088, 0.27372173, 0.21206113, 0.29010591,
                                  0.26843961, 0.38313593, 0.12846757, 0.4276979, 0.09602949]).reshape(-1, 1)
        coef = self.model.beta
        np.testing.assert_array_almost_equal(expected_coef, coef)

    def test_final_cost(self):
        expected_final_cost = 0.02674523805204521
        self.assertEqual(expected_final_cost, self.model.cost_history[-1])

    def test_predictions_shape(self):
        predictions = self.model.predict(self.x_test_std)
        self.assertTupleEqual(predictions.shape, (self.x_test_std.shape[1], 1))

    def test_train_score(self):
        expected_train_score = 0.973162160977947
        train_score = self.model.r_squared(self.x_train_std, self.y_train_std)
        self.assertAlmostEqual(expected_train_score, train_score)

    def test_test_score(self):
        expected_test_score = 0.9773369014894616
        test = self.model.r_squared(self.x_test_std, self.y_test_std)
        self.assertAlmostEqual(expected_test_score, test)
