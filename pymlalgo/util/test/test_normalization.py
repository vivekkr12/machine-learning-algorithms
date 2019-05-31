import unittest

import numpy as np

from pymlalgo.util.standardization import Standardizer


class NormalizerTest(unittest.TestCase):

    def setUp(self):
        self.n = 10
        self.d = 3
        self.features = np.arange(self.d * self.n).reshape(self.d, self.n)
        self.labels = np.arange(self.n).reshape(self.n, 1)

        self.features_mean_expected = np.mean(self.features, axis=1).reshape(self.d, 1)
        self.features_sd_expected = np.std(self.features, axis=1).reshape(self.d, 1) + 0.0001
        self.features_norm_expected = (self.features - self.features_mean_expected) / self.features_sd_expected

        self.labels_mean_expected = np.mean(self.labels, axis=0).reshape(1, 1)
        self.labels_sd_expected = np.std(self.labels, axis=0).reshape(1, 1) + 0.0001
        self.labels_norm_expected = (self.labels - self.labels_mean_expected) / self.labels_sd_expected

    def test_type_f(self):
        x_normalizer = Standardizer(self.features, type_='f')
        self.assertIsInstance(x_normalizer, Standardizer)

    def test_type_l(self):
        y_normalizer = Standardizer(self.labels, type_='l')
        self.assertIsInstance(y_normalizer, Standardizer)

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            Standardizer(self.features, type_='x')

    def test_features_std_shape(self):
        x_normalizer = Standardizer(self.features, type_='f')
        self.assertTupleEqual(x_normalizer.train_sd.shape, (self.d, 1))

    def test_labels_mean_shape(self):
        y_normalizer = Standardizer(self.labels, type_='l')
        self.assertTupleEqual(y_normalizer.train_mean.shape, (1, 1))

    def test_feature_normalization(self):
        x_normalizer = Standardizer(self.features, type_='f')
        features_norm = x_normalizer.standardize(self.features)
        self.assertTrue(np.array_equal(features_norm, self.features_norm_expected))

    def test_label_normalization(self):
        y_normalizer = Standardizer(self.labels, type_='l')
        labels_norm = y_normalizer.standardize(self.labels)
        self.assertTrue(np.array_equal(labels_norm, self.labels_norm_expected))

    def test_zero_std_normalization(self):
        data = np.ones(shape=(3, 10))
        normalizer = Standardizer(data, type_='f')
        self.assertFalse(np.any(normalizer.train_sd == 0.0))
