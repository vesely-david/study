import numpy as np


class Perceptron:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights = np.array([0] * n_inputs)
        self.bias = 0

    def train_gd(self, X, y, ny, n_iterations, log_step=100):
        for step in range(n_iterations):
            delta_w = -ny * self.grad_w(X, y)
            delta_b = -ny * self.grad_b(X, y)
            self.weights = self.weights + delta_w
            self.bias = self.bias + delta_b
            cost = self.cost(X, y)
            if step % log_step == 0:
                print(step, cost)

    def grad_wi(self, X, y, i):
        n = X.shape[0]
        return -(1. / n) * np.sum((y - self.predict_proba(X)) * self.predict_proba(X, True) * X[:, i])

    def grad_w(self, X, y):
        sum_deriv = [self.grad_wi(X, y, i) for i in range(X.shape[1])]
        return np.array(sum_deriv)

    def grad_b(self, X, y):
        n = X.shape[0]
        return -(1. / n) * np.sum((y - self.predict_proba(X)) * self.predict_proba(X, True))

    @staticmethod
    def sigmoid(y):
        return 1. / (1 + np.exp(-y))

    @classmethod
    def sigmoid_prime(cls, y):
        sigmoid = cls.sigmoid(y)
        return sigmoid * (1 - sigmoid)

    def predict_proba(self, X, derivative=False):
        y = np.matmul(X, self.weights.transpose()) + self.bias
        return self.sigmoid_prime(y) if derivative else self.sigmoid(y)

    def predict(self, X, threshold=.5):
        proba = self.predict_proba(X)
        proba[proba < threshold] = 0
        proba[proba >= threshold] = 1
        return proba

    def cost(self, X, y):
        n = X.shape[0]
        y_pred = self.predict_proba(X)
        return (1 / (2 * n)) * np.sum(np.square(y - y_pred))
