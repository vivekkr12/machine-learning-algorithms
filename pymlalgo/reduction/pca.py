import numpy as np


class PCA:
    POWER_ITERATION = 'power'
    OJA = 'oja'

    def __init__(self, x, method=OJA, eps=1e-8, learning_rate=1e-1, max_iter=1000, random_state=7):
        self.x = np.asarray(x)
        self.method = method
        self.eps = eps
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.n, self.d = x.shape

    def reduce_dimensions(self, d_reduced):
        if d_reduced > self.d:
            raise ValueError('d_reduced: {} cannot be greater than d: {}'.format(d_reduced, self.d))

        if self.method == PCA.POWER_ITERATION:
            return self.pca_power_iteration(d_reduced)
        elif self.method == PCA.OJA:
            return self.pca_oja(d_reduced)

    def power_iteration(self, a):
        np.random.seed(self.random_state)
        lambd = None
        v = np.random.randn(self.d, 1)
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
            a = a - np.asarray(lambd) * v.dot(v.T)
            x_pj = z.dot(v)

            x_reduced[:, j] = x_pj.flatten()

        return x_reduced

    def oja(self, a):
        np.random.seed(self.random_state)
        v = np.random.uniform(size=(self.d, 1))
        v = v / np.linalg.norm(v, ord=2)
        for itr in range(self.max_iter):
            v_last = v
            v = v + self.learning_rate * a.dot(v)
            v = v / np.linalg.norm(v, ord=2)

            if np.linalg.norm(v - v_last) < self.eps:
                break

        return v

    def pca_oja(self, d_reduced):
        x_reduced = np.zeros(shape=(self.n, d_reduced))

        z = self.x - np.mean(self.x, axis=0)
        a = z.T.dot(z)
        zj = z
        for j in range(d_reduced):
            v = self.oja(a)
            x_pj = z.dot(v)
            zj = zj - x_pj.dot(v.T)
            a = zj.T.dot(zj)
            x_reduced[:, j] = x_pj.flatten()

        return x_reduced
