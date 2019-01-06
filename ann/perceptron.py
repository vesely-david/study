import numpy as np


class Perceptron:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights = [0] * n_inputs
        self.bias = 0

    def learn_gd(self, X, y, ny, n_iterations):
        for step in range(n_iterations):
            delta_w = -ny * self.grad_w(X)
            delta_b = -ny * self.grad_b(X)
            self.weights = self.weights + delta_w
            self.bias = self.bias + delta_b
            cost = self.cost(X, y)
            print(step, cost)

    def grad_w(self, X):
        n = X.shape[0]
        sum_deriv = [(- 1. / (2. * n)) * np.sum(np.multiply(self.predict(X, True), X[i])) for i in X.shape[1]]
        return np.array(sum_deriv)

    def grad_b(self, X):
        n = X.shape[0]
        sum_deriv = [(- 1. / (2. * n)) * np.sum(self.predict(X, True)) for _ in X.shape[1]]
        return np.array(sum_deriv)

    @staticmethod
    def sigmoid(y):
        return 1. / (1 + np.exp(-y))

    @classmethod
    def sigmoid_prime(cls, y):
        sigmoid = cls.sigmoid(y)
        return sigmoid * (1 - sigmoid)

    def predict(self, X, deriv=False):
        y = X * self.weights + self.bias
        return self.sigmoid_prime(y) if deriv else self.sigmoid(y)

    def cost(self, X, y):
        # C = (1/2n) * Sum(||y-a||^2)
        n = X.shape[0]
        y_pred = self.predict(X)
        return (1 / (2 * n)) * np.sum(np.square(y - y_pred))
