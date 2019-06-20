import unittest

import numpy as np

from pymlalgo.util.standardization import Standardizer


class StandardizerTest(unittest.TestCase):

    def setUp(self):
        self.n = 10
        self.d = 3
        self.features = np.arange(self.n * self.d).reshape(self.n, self.d)
        self.labels = np.arange(self.n).reshape(self.n, 1)

        self.features_mean_expected = np.mean(self.features, axis=0).reshape(1, self.d)
        self.features_sd_expected = np.std(self.features, axis=0).reshape(1, self.d) + 0.0001
        self.features_standardized_expected = (self.features - self.features_mean_expected) / self.features_sd_expected

        self.labels_mean_expected = np.mean(self.labels, axis=0).reshape(1, 1)
        self.labels_sd_expected = np.std(self.labels, axis=0).reshape(1, 1) + 0.0001
        self.labels_standardized_expected = (self.labels - self.labels_mean_expected) / self.labels_sd_expected

    def test_type_features(self):
        x_standardizer = Standardizer(self.features)
        self.assertIsInstance(x_standardizer, Standardizer)

    def test_type_labels(self):
        y_standardizer = Standardizer(self.labels)
        self.assertIsInstance(y_standardizer, Standardizer)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            Standardizer(self.labels.flatten())

    def test_features_std_shape(self):
        x_normalizer = Standardizer(self.features)
        self.assertTupleEqual(x_normalizer.train_sd.shape, (1, self.d))

    def test_labels_mean_shape(self):
        y_standardizer = Standardizer(self.labels)
        self.assertTupleEqual(y_standardizer.train_mean.shape, (1, 1))

    def test_feature_standardization(self):
        x_standardizer = Standardizer(self.features)
        features_standardized = x_standardizer.standardize(self.features)
        self.assertTrue(np.array_equal(features_standardized, self.features_standardized_expected))

    def test_label_standardization(self):
        y_standardizer = Standardizer(self.labels)
        labels_standardized = y_standardizer.standardize(self.labels)
        self.assertTrue(np.array_equal(labels_standardized, self.labels_standardized_expected))

    def test_zero_std_standardization(self):
        data = np.ones(shape=(3, 10))
        standardizer = Standardizer(data)
        self.assertFalse(np.any(standardizer.train_sd == 0.0))
