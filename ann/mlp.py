import numpy as np

from utils.math import sigmoid


class MLP:
    def __init__(self, layers):
        self.biases = np.array([np.zeros(l) for l in layers])
        self.weights = np.array([np.zeros(shape=(layers[i], layers[i-1])) for i in range(1, len(layers))])

    def z_l(self, a_prev, l):
        return np.matmul(self.weights[l], a_prev) + self.biases[l]

    def a_l(self, a_prev, l):
        return sigmoid(self.z_l(a_prev, l))

