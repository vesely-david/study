import numpy as np



# sigmoid derivative:
# -(xi e^(-wi x - a2 x2 - b))/(e^(-a1 x1 - a2 x2 - b) + 1)^2

class Perceptron:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.weights = [0] * n_inputs
        self.bias = 0

    def learn_gd(self, X, y, ny, n_iterations):
        for step in range(n_iterations):
            delta_w = -ny * self.grad_w(X, y)
            delta_b = -ny * self.grad_b(X, y)
            self.weights = self.weights + delta_w
            self.bias = self.bias + delta_b
            cost = self.cost(X, y)
            print(step, cost)

    def grad_w(self, X, y):
        pass

    def grad_b(self, X, y):
        pass

    @staticmethod
    def sigmoid(y):
        return 1. / (1 + np.exp(-y))

    def predict(self, X):
        return self.sigmoid(X * self.weights + self.bias)

    def cost(self, X, y):
        # C = (1/2n) * Sum(||y-a||^2)
        n = X.shape[0]
        y_pred = self.predict(X)
        return (1 / (2 * n)) * np.sum(np.square(y - y_pred))
