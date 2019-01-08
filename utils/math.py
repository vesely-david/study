import numpy as np


def sigmoid(y):
    return 1. / (1 + np.exp(-y))


def sigmoid_prime(y):
    sigma = sigmoid(y)
    return sigma * (1 - sigma)
