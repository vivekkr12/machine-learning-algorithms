import unittest

import numpy as np

from pymlalgo.util.model_selection import train_test_split
from pymlalgo.regression.ridge_regression import RidgeRegression
from pymlalgo.util.standardization import Standardizer


class RidgeRegressionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n, cls.d = 1000, 10
        np.random.seed(7)

        x = np.random.uniform(0, 100, cls.n * cls.d).reshape(cls.n, cls.d)
        eps = np.random.normal(loc=0, scale=5, size=cls.n * cls.d).reshape(cls.n, cls.d)
        w = np.random.uniform(1, 5, cls.d).reshape(cls.d, 1)
        y = (x + eps).dot(w)

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=7, test_size=0.25)

        x_standardizer = Standardizer(x_train)
        cls.x_train_std = x_standardizer.standardize(x_train)
        cls.x_test_std = x_standardizer.standardize(x_test)

        y_standardizer = Standardizer(y_train)
        cls.y_train_std = y_standardizer.standardize(y_train)
        cls.y_test_std = y_standardizer.standardize(y_test)

        cls.model = RidgeRegression(cls.x_train_std, cls.y_train_std, lambd=0.1 / cls.n)
        cls.model.train()

    def test_coefficients(self):
        expected_weights = np.array([0.49164553, 0.32841494, 0.29913547, 0.24813988, 0.31602083,
                                     0.30158783, 0.41265551, 0.14537066, 0.4524806, 0.09626513]).reshape(-1, 1)
        weights = self.model.w
        np.testing.assert_array_almost_equal(expected_weights, weights, decimal=2)

    def test_final_cost(self):
        expected_final_cost = 0.03355478376080152
        self.assertAlmostEqual(expected_final_cost, self.model.cost_history[-1], places=4)

    def test_predictions_shape(self):
        predictions = self.model.predict(self.x_test_std)
        self.assertTupleEqual(predictions.shape, (self.x_test_std.shape[0], 1))

    def test_train_score(self):
        expected_train_score = 0.9665547873239119
        train_score = self.model.r_squared(self.x_train_std, self.y_train_std)
        self.assertAlmostEqual(expected_train_score, train_score, places=4)

    def test_test_score(self):
        expected_test_score = 0.9767454565709043
        test = self.model.r_squared(self.x_test_std, self.y_test_std)
        self.assertAlmostEqual(expected_test_score, test, places=4)

    def test_max_iter_exceeded(self):
        model = RidgeRegression(self.x_train_std, self.y_train_std, max_iter=10, backtracking_max_iter=10)
        model.train()
        self.assertEqual(model.n_iter, 10, 'Number of iterations must be 10')

    def test_train_with_init_weight(self):
        model = RidgeRegression(self.x_train_std, self.y_train_std, lambd=0.1 / self.n)
        model.train(self.model.w)
        # Since initial weight is from final trained model, n_iter must be very less
        self.assertLess(model.n_iter, self.model.n_iter)
