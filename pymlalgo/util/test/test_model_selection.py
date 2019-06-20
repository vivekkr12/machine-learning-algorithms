import unittest

import numpy as np

from pymlalgo.util.model_selection import train_test_split


class ModelSelectionTest(unittest.TestCase):

    def setUp(self):
        self.x = np.zeros(shape=(20, 3))
        self.y = np.ones(shape=(20, 1))

    def test_valid_shapes(self):
        test_size = 0.2
        train_n, test_n, d = 16, 4, 3

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size)

        self.assertTupleEqual(x_train.shape, (train_n, d), 'Shape of x_train did not match')
        self.assertTupleEqual(x_test.shape, (test_n, d), 'Shape of x_test did not match')
        self.assertTupleEqual(y_train.shape, (train_n, 1), 'Shape of y_train did not match')
        self.assertTupleEqual(y_test.shape, (test_n, 1), 'Shape of y_test did not match')

    def test_valid_shapes_floor(self):
        test_size = 0.28
        train_n, test_n, d = 15, 5, 3

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size)

        self.assertTupleEqual(x_train.shape, (train_n, d), 'Shape of x_train did not match')
        self.assertTupleEqual(x_test.shape, (test_n, d), 'Shape of x_test did not match')
        self.assertTupleEqual(y_train.shape, (train_n, 1), 'Shape of y_train did not match')
        self.assertTupleEqual(y_test.shape, (test_n, 1), 'Shape of y_test did not match')

    def test_one_d_shape_change(self):
        y = self.y.flatten()
        test_size = 0.2
        train_n, test_n, d = 16, 4, 3
        x_train, x_test, y_train, y_test = train_test_split(self.x, y, test_size=test_size)
        self.assertTupleEqual(y_train.shape, (train_n, 1), 'Shape of y_train did not change 2d')
        self.assertTupleEqual(y_test.shape, (test_n, 1), 'Shape of y_test did not change to 2d')
