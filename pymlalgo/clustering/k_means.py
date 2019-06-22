import logging

import numpy as np


class KMeans:

    RANDOM = 'random'
    K_MEANS_PP = 'kmeans++'

    def __init__(self, x, k, initialization=K_MEANS_PP, eps=0.001, max_iter=500, random_state=7):
        """
        Initializes an instance of Kmeans.

        Following instance attributes can be called after training the model.
        1. n_iter: Number of iterations it took to train the model
        2. labels: The cluster index assigned to each sample
        3. mu: The matrix of k centroids

        :param x: the feature matrix in which clusters have to be found
        :param k: number of clusters
        :param initialization: the method to choose first set of k centroids.
               Valid values are: ['random', 'kmeans++']. They can be provided
               using class level constants
        :param eps: stopping criteria - the minimum value of the magnitude of
               difference between two iterations of mu
        :param max_iter: maximum number of iterations
        :param random_state: seed to choose random sample for centroids
        """
        self.logger = logging.getLogger(__name__)

        self.n, self.d = x.shape
        self.x = x
        self.k = k
        self.initialization = initialization
        self.eps = eps
        self.max_iter = max_iter
        self.random_state = random_state

        self.n_iter = 0
        self.mu = np.zeros(shape=(k, self.d))
        self.labels = np.zeros(self.n)

    def get_cluster_data(self, ki, labels):
        """
        Subsets the input matrix x to data belonging to just one cluster
        :param ki: the index of the cluster, ki belongs to {1, 2, ..., k}
        :param labels: an array of cluster index assigned to each sample
        :return: samples from the given cluster
        """
        x_indexes = np.argwhere(labels == ki).flatten()
        x_cluster = self.x[x_indexes, :]
        return x_cluster

    def compute_cost(self, mu, labels):
        """
        Computes the value of the objective function which the algorithm is
        trying to minimize. Refer to theory for details
        :param mu: a vector of k centroids of shape (k, d)
        :param labels: an array of cluster index assigned to each sample
        :return: cost for the current
        """
        cost = 0
        for ki in range(self.k):
            x_cluster = self.get_cluster_data(ki, labels)
            distances = np.linalg.norm(x_cluster - mu[[ki]], axis=1)
            cluster_cost = np.sum(np.square(distances))
            cost += cluster_cost
        return cost

    def find_random_centroids(self):
        """
        Randomly selects k points in the sample as centroids
        """
        np.random.seed(self.random_state)
        arg_mu = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = self.x[arg_mu, :]

    def find_kpp_centroids(self):
        """
        Selects k points from the sample as centroids using the
        k-means++ algorithm
        """
        np.random.seed(self.random_state)
        arg_mu_1 = np.random.randint(low=0, high=self.n)
        mu_1 = self.x[arg_mu_1]
        self.mu[0] = mu_1

        mu_k = mu_1

        x_minus_mu = np.delete(self.x, arg_mu_1, axis=0)
        for ki in range(1, self.k):
            distances = np.linalg.norm(x_minus_mu - mu_k, axis=1)
            sum_squared_distances = np.sum(np.square(distances))
            probabilities = np.square(distances) / sum_squared_distances
            arg_mu_k = np.argmax(probabilities)
            mu_k = self.x[arg_mu_k]
            self.mu[ki] = mu_k
            x_minus_mu = np.delete(x_minus_mu, arg_mu_k, axis=0)

    def find_cluster(self, xi):
        """
        Finds cluster index for one sample
        :param xi: the sample
        :return: the cluster index
        """
        distances = np.square(np.linalg.norm(self.mu - xi, axis=1))
        xi_label = np.argmin(distances)
        return xi_label

    def k_means(self):
        """
        Runs the k-means algorithm in the sample data. The stopping criteria
        checks if the change in the means in each iterations is above a
        threshold. centroids must be initialized before calling this
        method.
        """
        cond = False
        itr = 0
        while cond is False and itr < self.max_iter:
            # assign each point to a cluster
            for i in range(self.n):
                label = self.find_cluster(self.x[[i]])
                self.labels[i] = label

            last_mu = np.copy(self.mu)

            # compute mean of each cluster
            for ki in range(self.k):
                x_cluster = self.get_cluster_data(ki, self.labels)
                self.mu[ki] = np.mean(x_cluster, axis=0)

            mean_change = self.mu - last_mu
            mean_change_magnitude = np.linalg.norm(mean_change)

            if mean_change_magnitude < self.eps:
                cond = True
                self.n_iter = itr + 1
            if itr == self.max_iter - 1:
                self.logger.warning('max iterations completed before stopping criteria reached')
                self.n_iter = itr + 1

            itr += 1

    def train(self, init_mu=None):
        """
        Trains the model using the parameters passed
        :param init_mu: If passed, it will be used as first set of
               centroids. Neither random or kmeans++ centroid search
               will run
        """
        if init_mu is not None:
            self.mu = init_mu
        elif KMeans.K_MEANS_PP == self.initialization:
            self.find_kpp_centroids()
        elif KMeans.RANDOM == self.initialization:
            self.find_random_centroids()
        self.k_means()
