import numpy as np


class PCA:
    POWER_ITERATION = 'power'
    OJA = 'oja'

    def __init__(self, x, method=OJA, eps=1e-3, max_iter=1000):
        self.x = x
        self.method = method
        self.eps = eps
        self.max_iter = max_iter
        self.n, self.d = x.shape

    def reduce_dimensions(self, d_reduced):
        if d_reduced > self.d:
            raise ValueError('d_reduced: {} cannot be greater than d: {}'.format(d_reduced, self.d))

        if self.method == PCA.POWER_ITERATION:
            pass
        elif self.method == PCA.OJA:
            pass

    def power_iteration(self, a):
        lambd = np.random.randn(1)
        v = np.random.randn(self.d , 1)
        for itr in range(self.max_iter):
            v = a.dot(v)
            v = v / np.linalg.norm(v, ord=2)
            lambd = v.T.dot(a).dot(v)

            error = a.dot(v) - lambd * v
            error_magnitude = np.linalg.norm(error)

            if error_magnitude < self.eps:
                break

        return lambd, v

    def pca_power_iteration(self, d_reduced):
        x_reduced = np.zeros(shape=(self.n, d_reduced))

        z = self.x - np.mean(self.x, axis=0)
        a = 1 / (self.n - 1) * z.T.dot(z)

        for j in range(d_reduced):
            lambd, v = self.power_iteration(a)
            a = a - lambd * v.dot(v.T)
            x_pj = z.dot(v)

            x_reduced[:, j] = x_pj

        return x_reduced

    def oja(self):
        pass
