import unittest
import numpy as np
from pymlalgo.util.model_selection import train_test_split
from pymlalgo.util.standardization import Standardizer
from pymlalgo.regression.regularised_logistic_regression import LogisticRegression


class MyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n, cls.d = 500, 10
        np.random.seed(7)

        x_class1 = np.random.normal(0, 5, cls.n * cls.d).reshape(cls.n, cls.d)
        x_class2 = np.random.normal(0, 10, cls.n * cls.d).reshape(cls.n, cls.d)
        w = np.random.uniform(1, 5, cls.d).reshape(cls.d, 1)
        y_class1 = np.full((cls.n, 1), -1)
        y_class2 = np.full((cls.n, 1), 1)

        x = np.concatenate((x_class1, x_class2))
        y = np.concatenate((y_class1, y_class2))

        x_train, x_test, cls.y_train, cls.y_test = train_test_split(x, y, random_state=7, test_size=0.25)

        x_standardizer = Standardizer(x_train)

        cls.x_train_std = x_standardizer.standardize(x_train)
        cls.x_test_std = x_standardizer.standardize(x_test)

        cls.model = LogisticRegression(cls.x_train_std, cls.y_train, lamd=1)
        cls.model.train()

    def test_coefficients(self):
        expected_weights = np.array([-0.01672295, 0.00731171, 0.00150423, -0.00112861, 0.00875998,
                                     -0.00206516, 0.00191913, 0.00345162, -0.00343227, 0.00824324]).reshape(-1, 1)
        weights = self.model.beta
        np.testing.assert_array_almost_equal(expected_weights, weights, decimal=2)

    def test_final_cost(self):
        expected_final_cost = 0.6925676658296561
        self.assertAlmostEqual(expected_final_cost, self.model.cost_history_fastgrad[-1], places=4)

    def test_predictions_shape(self):
        predictions = self.model.predict(self.x_test_std)
        self.assertTupleEqual(predictions.shape, (self.x_test_std.shape[0], 1))

    def test_train_score(self):
        expected_train_score = 0.5294117647058824
        train_score = self.model.score(self.x_train_std, self.y_train)
        self.assertAlmostEqual(expected_train_score, train_score, places=2)

    def test_test_score(self):
        expected_test_score = 0.48
        test = self.model.score(self.x_test_std, self.y_test)
        self.assertAlmostEqual(expected_test_score, test, places=2)


if __name__ == '__main__':
    unittest.main()
