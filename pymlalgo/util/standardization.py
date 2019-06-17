import numpy as np


class Standardizer:
    """
    Normalize the data to mean = 0 and standard deviation = 1.

    x = (x - x_mean) / x_std
    """

    def __init__(self, train, eps=0.0001):
        """
        :param train: train dataset from to calculate mean and sd
        :param eps: value of add to standard deviation to avoid divide by
                    0 errors
        """
        shape_data = train.shape
        if len(shape_data) == 1:
            raise ValueError('The shape of data passed is %s. Convert it to two dimensions by using '
                             'np.reshape(-1, 1)', shape_data)

        self.train = train
        self.train_mean = np.mean(self.train, axis=0).reshape(1, -1)
        self.train_sd = np.std(self.train, axis=0).reshape(1, -1) + eps

    def standardize(self, data):
        return (data - self.train_mean) / self.train_sd
