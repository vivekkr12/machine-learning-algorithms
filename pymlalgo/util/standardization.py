import numpy as np


class Standardizer:
    """
    Normalize the data to mean = 0 and standard deviation = 1.

    x = (x - x_mean) / x_std
    """

    FEATURES = 'f'
    LABELS = 'l'
    __VALID_TYPES__ = [FEATURES, LABELS]

    def __init__(self, train, type_, eps=0.0001):
        """
        :param train: train dataset from to calculate mean and sd
        :param type_: specify whether data is features or labels
                      accepted types are ['f', 'l'] for features
                      and labels respectively. Shape of features
                      must be (d, n) and shape of labels must be
                      (n, 1)
        :param eps: value of add to standard deviation to avoid divide by
                    0 errors
        """
        self.train = train

        if type_ not in Standardizer.__VALID_TYPES__:
            raise ValueError('type_ must be one of {}'.format(Standardizer.__VALID_TYPES__))

        axis = 0 if type_ == Standardizer.LABELS else 1
        self.train_mean = np.mean(self.train, axis=axis).reshape(-1, 1)
        self.train_sd = np.std(self.train, axis=axis).reshape(-1, 1) + eps

    def standardize(self, data):
        return (data - self.train_mean) / self.train_sd
