from unittest import TestCase

import numpy as np

from ann.perceptron import Perceptron
from utils.math import sigmoid, sigmoid_prime


class TestPerceptron(TestCase):
    def setUp(self):
        self.X = np.array([
            [1., 2., 3.],
            [3., 2., 1.]
        ])
        self.y = np.array([[0, 1]])
        self.clf = Perceptron(3)

    def test_grad_w(self):
        result_0 = self.clf.grad_w(self.X, self.y)
        expected_0 = [
            - sum([-.5 * sigmoid_prime(0) * 1, .5 * sigmoid_prime(0) * 3]) / 2,
            - sum([-.5 * sigmoid_prime(0) * 2, .5 * sigmoid_prime(0) * 2]) / 2,
            - sum([-.5 * sigmoid_prime(0) * 3, .5 * sigmoid_prime(0) * 1]) / 2
        ]
        self.assertListEqual(result_0.tolist(), expected_0)

    def test_grad_b(self):
        result_0 = self.clf.grad_b(self.X, self.y)
        expected_0 = - sum([
            -.5 * sigmoid_prime(0),
            .5 * sigmoid_prime(0)
        ]) / 2
        self.assertEqual(result_0, expected_0)

    def test_predict(self):
        y_pred = self.clf.predict_proba(self.X)
        expected = [sigmoid(0)] * 2
        self.assertListEqual(y_pred.tolist(), expected)

        y_pred_prime = self.clf.predict_proba(self.X, True)
        expected = [sigmoid_prime(0)] * 2
        self.assertListEqual(y_pred_prime.tolist(), expected)

    def test_cost(self):
        cost = self.clf.cost(self.X, self.y)
        predicted = self.clf.predict_proba(self.X)
        expected = sum([(0. - a) ** 2 for a in predicted]) / 4
        self.assertEqual(cost, expected)
