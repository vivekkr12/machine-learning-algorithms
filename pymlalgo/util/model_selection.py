import logging

import numpy as np

logger = logging.getLogger(__name__)


def train_test_split(x, y, test_size=0.25, random_state=7):
    """
    Splits the data into train and test sets

    :param x: features matrix - a numpy array of shape (n, d)
    :param y: labels matrix - numpy array of shape (n, 1)
    :param test_size: fraction of data in the test set - value range from (0, 1)
    :param random_state: seed to generate random indexes for data split
    :return: a tuple of (x_train, x_test, y_train, y_test)
    """

    shape_y = y.shape
    if len(shape_y) == 1:
        y = y.reshape(-1, 1)
        logger.warning('The shape of y passed is %s but it will be converted to %s\n'
                       'It is recommended to reshape y using np.reshape(-1, 1)', shape_y, y.shape)

    np.random.seed(random_state)
    n, d = x.shape
    test_size = np.floor(test_size * n).astype(int)
    test_indices = np.random.randint(low=0, high=n, size=test_size)

    x_test = x[test_indices, :]
    y_test = y[test_indices, :]
    x_train = np.delete(x, test_indices, axis=0)
    y_train = np.delete(y, test_indices, axis=0)

    return x_train, x_test, y_train, y_test
