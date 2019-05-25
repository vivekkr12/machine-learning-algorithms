import numpy as np


def train_test_split(x, y, test_size=0.25, random_state=7):
    """
    Splits the data into train and test sets

    :param x: features matrix of shape (d, n)
    :param y: labels matrix of shape (n, 1)
    :param test_size: fraction of data in the test set - value range from (0, 1)
    :param random_state: seed to generate random indexes for data split
    :return: a tuple of (x_train, x_test, y_train, y_test)
    """
    np.random.seed(random_state)
    d, n = x.shape
    test_size = np.floor(test_size * n).astype(int)
    test_indices = np.random.randint(low=0, high=n, size=test_size)

    x_test = x[:, test_indices]
    y_test = y[test_indices, :]
    x_train = np.delete(x, test_indices, axis=1)
    y_train = np.delete(y, test_indices, axis=0)

    return x_train, x_test, y_train, y_test
