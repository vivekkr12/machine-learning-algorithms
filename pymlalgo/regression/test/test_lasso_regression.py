import unittest
import numpy as np
from pymlalgo.util.standardization import Standardizer
from pymlalgo.util.model_selection import train_test_split
from pymlalgo.regression.lasso_regression import LassoRegression


class LassoRegressionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n, cls.d = 1000, 10
        np.random.seed(7)

        x = np.random.uniform(0, 50, cls.n * cls.d).reshape(cls.n, cls.d)
        eps = np.random.normal(loc=0, scale=2, size=cls.n * cls.d).reshape(cls.n, cls.d)
        w = np.random.uniform(1, 5, cls.d).reshape(cls.d, 1)
        y = (x + eps).dot(w)

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=7, test_size=0.25)

        x_standardizer = Standardizer(x_train)
        cls.x_train_std = x_standardizer.standardize(x_train)
        cls.x_test_std = x_standardizer.standardize(x_test)

        y_standardizer = Standardizer(y_train)
        cls.y_train_std = y_standardizer.standardize(y_train)
        cls.y_test_std = y_standardizer.standardize(y_test)

        cls.model = LassoRegression(cls.x_train_std, cls.y_train_std, lambd=0.5 / 2)
        cls.model.cycliccoorddescent()
        cls.model.randcoorddescent()

    def test_coefficients_cyclic(self):
        expected_weights = np.array([0.3346511, 0.18797564, 0.1742291, 0.10907644, 0.13760524,
                                     0.14054056, 0.27984296, 0, 0.32117783, 0]).reshape(-1, 1)
        weights = self.model.cyclic_beta
        np.testing.assert_array_almost_equal(expected_weights, weights, decimal=2)

    def test_coefficients_random(self):
        expected_weights = np.array([0.3346511, 0.18797564, 0.1742291, 0.10907644, 0.13760524,
                                     0.14054056, 0.27984296, 0, 0.32117783, 0]).reshape(-1, 1)
        weights = self.model.random_beta
        np.testing.assert_array_almost_equal(expected_weights, weights, decimal=2)

    def test_final_cost_cyclic(self):
        expected_final_cost = 0.6197832659021694
        self.assertAlmostEqual(expected_final_cost, self.model.cost_history_cyclic[-1], places=4)

    def test_final_cost_random(self):
        expected_final_cost = 0.6197832659021695
        self.assertAlmostEqual(expected_final_cost, self.model.cost_history_rand[-1], places=4)


if __name__ == '__main__':
    unittest.main()
