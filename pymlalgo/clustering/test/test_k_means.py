import unittest
from unittest.mock import MagicMock

import numpy as np

from pymlalgo.clustering.k_means import KMeans


class KMeansTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(7)
        cls.n, cls.d = 200, 2
        cls.k = 4
        x1 = np.random.normal(loc=1, scale=1, size=cls.n // cls.k * cls.d).reshape(cls.n // cls.k, cls.d)
        x2 = np.random.normal(loc=5, scale=1, size=cls.n // cls.k * cls.d).reshape(cls.n // cls.k, cls.d)
        x3 = np.random.normal(loc=10, scale=1, size=cls.n // cls.k * cls.d).reshape(cls.n // cls.k, cls.d)
        x4 = np.random.normal(loc=15, scale=1, size=cls.n // cls.k * cls.d).reshape(cls.n // cls.k, cls.d)

        cls.x = np.concatenate((x1, x2, x3, x4), axis=0)

    def test_train_with_random(self):
        k_means_random = KMeans(self.x, self.k, initialization=KMeans.RANDOM, eps=1e-5)
        k_means_random.train()
        self.assertEqual(len(np.unique(k_means_random.labels)), self.k)

    def test_train_with_kpp(self):
        k_means_kpp = KMeans(self.x, self.k, initialization=KMeans.K_MEANS_PP, eps=1e-5)
        k_means_kpp.train()
        self.assertEqual(len(np.unique(k_means_kpp.labels)), self.k)

    def test_get_cluster_data(self):
        labels = np.zeros(self.n)
        labels[np.random.randint(low=0, high=self.n, size=50)] = 1

        expected_values = self.x[labels == 1]
        k_means_kpp = KMeans(self.x, self.k, initialization=KMeans.K_MEANS_PP, eps=1e-5)
        k_means_kpp.train()
        x_cluster = k_means_kpp.get_cluster_data(1, labels)

        np.testing.assert_array_equal(x_cluster, expected_values)

    def test_compute_cost(self):
        x = np.array([1, 2, 9, 10]).reshape(-1, 1)
        labels = np.array([0, 0, 1, 1])
        mu = np.array([1.5, 9.5]).reshape(-1, 1)
        k = 2
        model = KMeans(x, k)
        cost = model.compute_cost(mu, labels)

        self.assertEqual(cost, 1)

    def test_find_random_centroid(self):
        np.random.seed(0)
        random_indexes = np.random.randint(low=0, high=self.n, size=self.k)

        expected_centroids = self.x[random_indexes, :]
        model = KMeans(self.x, self.k, random_state=0)
        model.find_random_centroids()

        np.testing.assert_array_equal(expected_centroids, model.mu)

    def test_find_kpp_centroid(self):
        expected_centroids = np.array([[13.87626757, 14.41023753],
                                       [-0.7083392, -0.80309866],
                                       [15.42080098, 15.77197743],
                                       [0.59477214, -1.2883151]])

        model = KMeans(self.x, self.k, random_state=0)
        model.find_kpp_centroids()

        np.testing.assert_allclose(model.mu, expected_centroids)

    def test_find_cluster(self):
        x = np.array([1, 2, 9, 10]).reshape(-1, 1)
        model = KMeans(x, 2)
        model.mu = np.array([3, 10]).reshape(-1, 1)
        cluster_index = model.find_cluster([[1]])

        self.assertEqual(cluster_index, 0)

    # noinspection PyUnresolvedReferences
    def test_train_with_init_mu(self):
        init_mu = np.array([[1, 1],
                            [5, 5],
                            [10, 10],
                            [15, 15]])
        model = KMeans(self.x, self.k)
        model.find_kpp_centroids = MagicMock()
        model.find_random_centroids = MagicMock()
        model.train(init_mu)

        self.assertFalse(model.find_kpp_centroids.called)
        self.assertFalse(model.find_random_centroids.called)

    def test_train_max_error_exceeded(self):
        model = KMeans(self.x, self.k, max_iter=2)
        model.train()
        self.assertEqual(model.n_iter, 2)
