import unittest

import numpy as np

from pymlalgo.util.model_selection import train_test_split


class ModelSelectionTest(unittest.TestCase):

    def setUp(self):
        self.x = np.zeros(shape=(3, 20))
        self.y = np.ones(shape=(20, 1))

    def test_valid_shapes(self):
        test_size = 0.2
        train_n, test_n, d = 16, 4, 3

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size)

        self.assertTupleEqual(x_train.shape, (d, train_n), 'Shape of x_train did not match')
        self.assertTupleEqual(x_test.shape, (d, test_n), 'Shape of x_test did not match')
        self.assertTupleEqual(y_train.shape, (train_n, 1), 'Shape of y_train did not match')
        self.assertTupleEqual(y_test.shape, (test_n, 1), 'Shape of y_test did not match')

    def test_valid_shapes_floor(self):
        test_size = 0.28
        train_n, test_n, d = 15, 5, 3

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size)

        self.assertTupleEqual(x_train.shape, (d, train_n), 'Shape of x_train did not match')
        self.assertTupleEqual(x_test.shape, (d, test_n), 'Shape of x_test did not match')
        self.assertTupleEqual(y_train.shape, (train_n, 1), 'Shape of y_train did not match')
        self.assertTupleEqual(y_test.shape, (test_n, 1), 'Shape of y_test did not match')
