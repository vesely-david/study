from unittest import TestCase

import numpy as np
from math import exp

from ann.perceptron import Perceptron


class TestPerceptron(TestCase):
    def test_grad_w(self):
        self.fail()

    def test_grad_b(self):
        # def grad(X, bias, weights)
        #
        # X = np.array([
        #     [1., 2., 3.],
        #     [3., 2., 1.]
        # ])


    def test_sigmoid(self):
        subjects = np.array([-2, -1, 0, 1])
        expected = np.array([.1192029220221175559403, .2689414213699951207488, .5, .7310585786300048792512])
        result = Perceptron.sigmoid(subjects)
        self.assertEqual(type(result), np.ndarray)
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)

    def test_sigmoid_prime(self):
        def sp(x):
            return exp(-x) / ((1 + exp(-x)) ** 2)

        subjects = np.array([-2, -1, 0, 1])
        expected = np.array([sp(x) for x in subjects])
        result = Perceptron.sigmoid_prime(subjects)
        self.assertEqual(type(result), np.ndarray)
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e)

    def test_predict(self):
        self.fail()

    def test_cost(self):
        self.fail()
