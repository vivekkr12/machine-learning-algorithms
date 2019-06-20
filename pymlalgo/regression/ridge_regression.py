import logging

import numpy as np


class RidgeRegression:
    """
    Linear regression using least squares loss with L2 regularization.
    """

    def __init__(self, x_train, y_train, lambd=0.1, min_grad=0.001, max_iter=1000, backtracking_max_iter=500):
        """
        Initializes and instance of RidgeRegression

        :param x_train: training features of shape (n, d)
        :param y_train: training labels of shape (n, 1)
        :param lambd: regularization parameter, default value 0.1
        :param min_grad: minimum slope to stop gradient descent, default - 0.001
        :param max_iter: maximum number if iteration - default 1000. If minimum gradient is not
                         is not reached in max iteration, the algorithm will stop optimizing further
        :param backtracking_max_iter: maximum number of iterations to optimize learning rate using
                                      backtracking line search
        """
        self.logger = logging.getLogger(__name__)

        self.n_train, self.d = x_train.shape

        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)
        self.lambd = lambd
        self.min_grad = min_grad
        self.max_iter = max_iter
        self.backtracking_max_iter = backtracking_max_iter

        # arrays to contain costs, coefficients, and gradients after each iteration
        self.cost_history = np.zeros(max_iter)
        self.grad_magnitude_history = np.zeros(max_iter)
        self.w_history = np.zeros((self.d, max_iter))

        # initialize to an arbitrary learning rate
        self.init_learning_rate = 0.001
        self.w = None

    def __compute_initial_learning_rate__(self):
        """
        Computes the initial learning rate by calculating the Lipschitz. Backtracking
        line search optimized this learning rate

        :return: initial learning rate
        """
        eigen_values = np.real(np.linalg.eigvals(np.cov(self.x_train.T)))
        lipschitz = np.max(eigen_values) + self.lambd
        initial_learning_rate = 1 / lipschitz
        return initial_learning_rate

    def backtracking(self, w, grad, grad_magnitude, cost, alpha=0.5, gamma=0.8):
        """
        Computes optimal learning rate for each iteration.
        alpha and gamma are constants and the default values
        had proven to work well in most scenarios.

        Parameters grad, grad_magnitude and cost are passed instead of being
        calculated in the method because they are already calculated in during
        gradient descent

        :param w: weight vector after last iteration
        :param grad: grad at the given beta
        :param grad_magnitude: grad magnitude at the given beta
        :param cost: cost at the given beta
        :param alpha: a constant - default 0.5
        :param gamma: a constant - default 0.8
        :return: optimal learning rate
        """
        learning_rate = self.init_learning_rate
        condition = False
        i = 0  # Iteration counter
        while condition is False and i < self.backtracking_max_iter:
            lhs = self.compute_cost(w - learning_rate * grad)
            rhs = cost - alpha * learning_rate * grad_magnitude ** 2
            if lhs <= rhs:
                condition = True
            elif i == self.backtracking_max_iter - 1:
                self.logger.warning('maximum number of iterations for backtracking reached')
                break
            else:
                learning_rate *= gamma
                i += 1
        return learning_rate

    def compute_cost(self, w):
        """
        Computes the cost (value of objective function) for the given coefficient

        :param w: weight vector
        :return: cost
        """
        residuals = self.y_train - self.x_train.dot(w)
        least_squares = np.square(np.linalg.norm(residuals, ord=2))
        regularization = self.lambd * np.square(np.linalg.norm(w, ord=2))
        return (1 / self.n_train) * least_squares + regularization

    def compute_grad(self, w):
        """
        Computes the gradient for the given coefficient

        :param w: weight vector
        :return: grad
        """
        residuals = self.y_train - self.x_train.dot(w)
        least_square_grad = (-2 / self.n_train) * self.x_train.T.dot(residuals)
        reg_grad = 2 * self.lambd * w
        return least_square_grad + reg_grad

    def fast_gradient_descent(self, init_w=None):
        """
        Runs accelerated gradient descent algorithm to minimize the cost. The stopping criteria is
        minimum gradient or maximum number of iterations whichever comes earlier

        :param init_w: by default the weights will be initialized to 0. However, if coefficients are known
                          from a similar problem they can be used to make convergence much faster.
        """
        if init_w is None:
            beta = np.zeros((self.d, 1))
            theta = np.zeros((self.d, 1))
        else:
            beta = init_w
            theta = init_w

        condition = False
        i = 0

        while condition is False:

            grad_theta = self.compute_grad(beta)
            cost = self.compute_cost(beta)
            grad_magnitude = np.linalg.norm(grad_theta, ord=2)
            learning_rate = self.backtracking(beta, grad_theta, grad_magnitude, cost)
            self.w_history[:, i] = beta.flatten()
            self.cost_history[i] = cost
            self.grad_magnitude_history[i] = grad_magnitude

            if grad_magnitude < self.min_grad:
                condition = True
            elif i == self.max_iter - 1:
                self.logger.warning('max iteration for fast gradient descent reached before condition became true')
                condition = True
            else:
                i += 1
                beta = theta - learning_rate * grad_theta
                theta = beta + i / (i + 3) * (beta - self.w_history[:, (i - 1)].reshape(self.d, 1))

        self.w_history = self.w_history[:, 0:i]
        self.cost_history = self.cost_history[0:i]
        self.grad_magnitude_history = self.grad_magnitude_history[0:i]
        self.w = beta

    def train(self, init_w=None):
        """
        Trains the model and optimizes the coefficients

        :param init_w: by default the coefficients will be initialized to 0. However, if coefficients are known
                          from a similar problem they can be used to make convergence much faster.
        """
        self.init_learning_rate = self.__compute_initial_learning_rate__()
        self.fast_gradient_descent(init_w)

    def predict(self, x, w=None):
        """
        Make predictions

        :param x: the features dataset of shape (n, d)
        :param w: the coefficients. default - None. If not passed, it will use the coefficients
                     optimized during training
        :return: the predictions in shape (n, 1)
        """
        if w is None:
            w = self.w

        return x.dot(w)

    def r_squared(self, x, y, w=None):
        """
        Calculates the R Squared value (coefficient of determination). The maximum value is 1 and
        higher values indicate better accuracy of the model.

        :param x: features dataset
        :param y: labels dataset
        :param w: coefficients
        :return: r squared
        """
        predictions = self.predict(x, w)
        total_sq_error = np.sum(np.square(y - predictions))
        total_variation_y = np.sum(np.square(y - np.mean(y)))
        r_squared = 1 - total_sq_error / total_variation_y
        return r_squared
