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
        expected_coef = np.array([0.4590115, 0.28894745, 0.27342446, 0.21193167, 0.29020445,
                                 0.26846985, 0.38304797, 0.12870089, 0.42763333, 0.0962281]).reshape(-1, 1)
        coef = self.model.beta
        np.testing.assert_array_almost_equal(expected_coef, coef, decimal=4)

    def test_final_cost(self):
        expected_final_cost = 0.026745657020640256
        self.assertAlmostEqual(expected_final_cost, self.model.cost_history[-1], places=4)

    def test_predictions_shape(self):
        predictions = self.model.predict(self.x_test_std)
        self.assertTupleEqual(predictions.shape, (self.x_test_std.shape[1], 1))

    def test_train_score(self):
        expected_train_score = 0.9731618072817436
        train_score = self.model.r_squared(self.x_train_std, self.y_train_std)
        self.assertAlmostEqual(expected_train_score, train_score, places=4)

    def test_test_score(self):
        expected_test_score = 0.9773325023618751
        test = self.model.r_squared(self.x_test_std, self.y_test_std)
        self.assertAlmostEqual(expected_test_score, test, places=4)
