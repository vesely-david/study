from unittest import TestCase

import numpy as np
from math import exp

from ann.perceptron import Perceptron


def sp(x):
    return exp(-x) / ((1 + exp(-x)) ** 2)


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
            sum([-.5 * Perceptron.sigmoid_prime(0) * 1, .5 * Perceptron.sigmoid_prime(0) * 3]) / 2,
            sum([-.5 * Perceptron.sigmoid_prime(0) * 2, .5 * Perceptron.sigmoid_prime(0) * 2]) / 2,
            sum([-.5 * Perceptron.sigmoid_prime(0) * 3, .5 * Perceptron.sigmoid_prime(0) * 1]) / 2
        ]
        self.assertListEqual(result_0.tolist(), expected_0)

    def test_grad_b(self):
        result_0 = self.clf.grad_b(self.X, self.y)
        expected_0 = sum([
            -.5 * Perceptron.sigmoid_prime(0),
            .5 * Perceptron.sigmoid_prime(0)
        ]) / 2
        self.assertEqual(result_0, expected_0)

    def test_sigmoid(self):
        subjects = np.array([-2, -1, 0, 1])
        expected = np.array([.1192029220221175559403, .2689414213699951207488, .5, .7310585786300048792512])
        result = Perceptron.sigmoid(subjects)
        self.assertEqual(type(result), np.ndarray)
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)

    def test_sigmoid_prime(self):
        subjects = np.array([-2, -1, 0, 1])
        expected = np.array([sp(x) for x in subjects])
        result = Perceptron.sigmoid_prime(subjects)
        self.assertEqual(type(result), np.ndarray)
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)

    def test_predict(self):
        y_pred = self.clf.predict(self.X)
        expected = [Perceptron.sigmoid(0)] * 2
        self.assertListEqual(y_pred.tolist(), expected)

        y_pred_prime = self.clf.predict(self.X, True)
        expected = [Perceptron.sigmoid_prime(0)] * 2
        self.assertListEqual(y_pred_prime.tolist(), expected)

    def test_cost(self):
        cost = self.clf.cost(self.X, self.y)
        predicted = self.clf.predict(self.X)
        expected = sum([(0. - a) ** 2 for a in predicted]) / 4
        self.assertEqual(cost, expected)
