import numpy as np


class LassoRegression:

    def __init__(self, x_train, y_train, lambd, max_iter=1000):
        """
        Initializes and creates a instance of l1 regularised linear regression

        :param x_train: Training features of shape(nxd)
        :param y_train: Training labels (nx1)
        :param lambd: l1 regularization penalty
        :param max_iter: max_iterations
        """

        self.x_train = x_train
        self.y_train = y_train
        self.lambd = lambd
        self.max_iter = max_iter
        self.n, self.d = self.x_train.shape

    def computeobj(self, beta):
        """
        Computes the objective value of lasso problem

        :param beta:coefficients of d features (dx1)
        :return: objective value to be minimized
        """
        a = np.square(np.linalg.norm(self.y_train - self.x_train.dot(beta), ord=2))
        b = self.lambd * np.linalg.norm(beta, ord=1)
        return (a / self.n + b).squeeze()

    def compute_beta_j(self, j, beta):

        """
        Updates the jth beta according to the condition

        :param j: jth coefficient to be updated
        :param beta: coefficient vector (d x 1)
        :return: updated jth coefficient as per the conditions
        """

        x_minus_j = np.delete(self.x_train, j, axis=1)

        beta_minus_j = np.delete(beta, j, axis=0).reshape(self.d - 1, 1)

        r_minus_j = self.y_train - (x_minus_j.dot(beta_minus_j))

        x_j = self.x_train[:, j].reshape(self.n, 1)
        z_j = np.square(np.linalg.norm(x_j, ord=2))

        condition = (2 / self.n) * x_j.T.dot(r_minus_j)
        condition = condition.squeeze()

        den = ((2 / self.n) * z_j).squeeze()

        if condition <= -self.lambd:
            num = (self.lambd + condition)
            beta_j = num / den
            return beta_j

        elif condition >= self.lambd:
            num = (-self.lambd + condition)
            beta_j = num / den
            return beta_j

        elif np.abs(condition) <= self.lambd:
            beta_j = 0
            return beta_j

    def cycliccoorddescent(self):
        """
        updates the coefficients in a cyclic fashion

        :return: updated coefficients in a cyclic manner
        """
        init_beta = np.zeros((self.d, 1))
        beta = init_beta
        self.beta_history_cyclic = np.array(beta)
        self.cost_history_cyclic = []
        for it in range(self.max_iter):
            for j in range(self.d):
                beta[j, 0] = self.compute_beta_j(j, beta)
            self.cost_history_cyclic.append(self.computeobj(beta))
            self.beta_history_cyclic = np.append(self.beta_history_cyclic, beta, axis=1)

        self.cyclic_beta = beta

    def pickcord(self):
        """
        generates a random number between 1 to d

        :return: random number
        """
        return np.random.randint(low=0, high=self.d, size=1)

    def randcoorddescent(self):
        """
        updates a random coefficient(from pickcord)

        :return: updated coefficient
        """

        init_beta = np.zeros((self.d, 1))
        beta = init_beta
        self.cost_history_rand = []
        self.beta_history_random = np.array(beta)
        for it in range(self.max_iter * self.d):
            j = self.pickcord()
            beta[j] = self.compute_beta_j(j, beta)
            if it % self.d == 0:
                if it % self.d == 0:
                    self.cost_history_rand.append(self.computeobj(beta))
                    self.beta_history_random = np.append(self.beta_history_random, beta, axis=1)
        self.random_beta = beta
