import numpy as np
import logging


class LogisticRegression:

    def __init__(self, x_train, y_train, lamd, eps=0.0001, max_iterations=1000):

        """
        Initializes and creates a instance of l2 regularised logistic regression

        :param x_train: training features of shape (n, d)
        :param y_train: training labels of shape (n, 1)
        :param lamd: regularization parameter, default value 0.1
        :param eps: minimum slope to stop gradient descent, default - 0.001
        :param max_iterations: maximum number if iteration - default 1000. If minimum gradient is not
                               is not reached in max iteration, the algorithm will stop optimizing further

        """

        self.x_train = x_train
        self.y_train = y_train
        self.lamd = lamd
        self.max_iterations = max_iterations
        self.eps = eps
        self.n_train, self.d_train = self.x_train.shape

    def train(self):
        """
        Trains the model and optimizes the coefficients

        """

        self.init_learning_rate = self.__compute_init_learning_rate__()
        self.beta, self.cost_history_fastgrad, self.beta_history = self.fast_grad_descent()

    def obj(self, beta):
        """
        Computes the cost (value of objective function) for the given coefficient

        :param beta: weight vector for logistic regression
        :return: cost
        """

        a = np.exp(- self.y_train * (self.x_train.dot(beta)))
        b = np.sum(np.log(1 + a))
        residuals = self.lamd * np.square(np.linalg.norm(beta))
        cost = (b / self.n_train + residuals)
        return cost.squeeze()

    def compute_grad(self, beta):
        """
        Computes the gradient for the given coefficient

        :param beta: weight vector
        :return: grad
        """
        x = self.y_train * (self.x_train.dot(beta))
        a = (1 / (1 + np.exp(x)))
        p = np.diag(np.array(a).reshape(self.n_train, ))
        grad = -1 / self.n_train * (self.x_train.T.dot(p).dot(self.y_train)) + 2 * self.lamd * beta
        return grad

    def __compute_init_learning_rate__(self):
        """
        Computes the initial learning rate by calculating the Lipschitz. Backtracking
        line search optimized this learning rate
        :return: initial learning rate
        """
        eigenvalues, eigenvectors = np.linalg.eigh(1 / self.n_train * self.x_train.dot(self.x_train.T))
        lipschitz = max(eigenvalues) + self.lamd
        return 1 / lipschitz

    def bt_line_search(self, beta, alpha=0.5, gamma=0.8, max_iter=100):
        """
        Computes optimal learning rate for each iteration.
        alpha and gamma are constants and the default values
        had proven to work well in most scenarios.

        Parameters grad, grad_magnitude and cost are passed instead of being
        calculated in the method because they are already calculated in during
        gradient descent
        :param beta:weight vector after last iteration
        :param alpha:a constant - default 0.5
        :param gamma:a constant - default 0.8
        :param max_iter:maximum iterations for each call to bt_line_search method
        :return: optimal learning rate
        """
        learning_rate = self.init_learning_rate
        grad = self.compute_grad(beta)
        z = beta - learning_rate * grad
        lhs = self.obj(z)
        rhs = self.obj(beta) - alpha * learning_rate * np.square(np.linalg.norm(grad))
        i = 0
        while rhs < lhs and i < max_iter:
            learning_rate *= gamma
            z = beta - learning_rate * grad
            lhs = self.obj(z)
            rhs = self.obj(beta) - alpha * learning_rate * np.square(np.linalg.norm(grad))
            i += 1
        return learning_rate

    def fast_grad_descent(self):
        """
        Runs accelerated gradient descent algorithm to minimize the cost. The stopping criteria is
        minimum gradient or maximum number of iterations whichever comes earlier.
        """

        beta = np.zeros((self.d_train, 1))
        theta = np.zeros((self.d_train, 1))
        cost_history_fastgrad = []
        beta_history = np.array(beta)

        for it in range(self.max_iterations):
            theta_grad = self.compute_grad(theta)
            error = np.linalg.norm(theta_grad)
            if error > self.eps:
                cost_history_fastgrad.append(self.obj(beta))
                t = self.bt_line_search(beta)
                beta = theta - (t * theta_grad)
                theta = beta + it / (it + 3) * (beta - beta_history[:, (it)].reshape(self.d_train, 1))
                beta_history = np.append(beta_history, beta, axis=1)
            else:
                break

        return beta, cost_history_fastgrad, beta_history

    def predict(self, x):
        """
        Make predictions

        :param x: the features dataset of shape (n, d)
        :return: the predictions in shape (n, 1)
        """
        pred = 1 / (1 + np.exp(-x.dot(self.beta))) > 0.5
        predictions = pred * 2 - 1
        return predictions

    def score(self, x, y):
        """
        Calulates mean accuracy

        :param x: the features dataset of shape (n, d)
        :param y: label dataset of shape (n,1)
        :return: mis classification error
        """
        predictions = self.predict(x)
        error = np.mean(predictions == y)
        return error
