import numpy as np


class Perceptron:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights = np.array([0] * n_inputs)
        self.bias = 0

    def learn_gd(self, X, y, ny, n_iterations):
        for step in range(n_iterations):
            delta_w = -ny * self.grad_w(X)
            delta_b = -ny * self.grad_b(X)
            self.weights = self.weights + delta_w
            self.bias = self.bias + delta_b
            cost = self.cost(X, y)
            print(step, cost)

    def grad_wi(self, X, y, i):
        n = X.shape[0]
        return (1. / n) * np.sum((y - self.predict(X)) * self.predict(X, True) * X[:, i])

    def grad_w(self, X, y):
        sum_deriv = [self.grad_wi(X, y, i) for i in range(X.shape[1])]
        return np.array(sum_deriv)

    def grad_b(self, X, y):
        n = X.shape[0]
        return (1. / n) * np.sum((y - self.predict(X)) * self.predict(X, True))

    @staticmethod
    def sigmoid(y):
        return 1. / (1 + np.exp(-y))

    @classmethod
    def sigmoid_prime(cls, y):
        sigmoid = cls.sigmoid(y)
        return sigmoid * (1 - sigmoid)

    def predict(self, X, derivative=False):
        y = np.matmul(X, self.weights.transpose()) + self.bias
        return self.sigmoid_prime(y) if derivative else self.sigmoid(y)

    def cost(self, X, y):
        # C = (1/2n) * Sum(||y-a||^2)
        n = X.shape[0]
        y_pred = self.predict(X)
        return (1 / (2 * n)) * np.sum(np.square(y - y_pred))
