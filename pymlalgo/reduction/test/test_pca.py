import unittest

import numpy as np

from pymlalgo.reduction.pca import PCA


class PCATest(unittest.TestCase):

    @staticmethod
    def __generate_simulated_data__(n, d, k, random_state=7):
        np.random.seed(random_state)
        n_class = n // k
        sim_data = np.zeros(shape=(n, d))
        for i in range(k):
            start = i
            end = i + 1 + np.random.rand()
            x_k = np.random.uniform(start, end, n_class * d).reshape(n_class, d)
            sim_data[i * k: i * k + n_class, :] = x_k
        return np.asarray(sim_data)

    @classmethod
    def setUpClass(cls):
        cls.x = PCATest.__generate_simulated_data__(100, 50, 5)

    def test_power_iteration_oja_similar(self):
        pca_pi = PCA(x=self.x, method=PCA.POWER_ITERATION)
        x_reduced_pi = pca_pi.reduce_dimensions(4)

        pca_oja = PCA(x=self.x, method=PCA.OJA)
        x_reduced_oja = pca_oja.reduce_dimensions(4)

        np.testing.assert_allclose(np.abs(x_reduced_pi), np.abs(x_reduced_oja), rtol=1e-1, atol=1e-1)

    def test_error_on_d_reduced_greater(self):
        pca_oja = PCA(x=self.x, method=PCA.OJA)

        with self.assertRaises(ValueError):
            pca_oja.reduce_dimensions(60)
