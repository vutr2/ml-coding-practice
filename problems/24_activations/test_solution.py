import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


X = np.array([-3.0, -0.5, 0.0, 0.7, 2.0])
XN = np.array([-3.0, -0.5, 0.7, 2.0])   # tránh điểm gãy 0 khi so gradient số


def _num(f, x):
    return np.array([numerical_grad(lambda v: f(v)[0], np.array([xi])) for xi in x]).ravel()


def test_relu():
    """relu và relu_grad"""
    close(sol.relu(X), [0, 0, 0, 0.7, 2.0], "relu")
    close(sol.relu_grad(X), [0, 0, 0, 1, 1], "relu_grad")


def test_leaky_relu():
    """leaky_relu và leaky_relu_grad (alpha = 0.1)"""
    close(sol.leaky_relu(X, 0.1), [-0.3, -0.05, 0, 0.7, 2.0], "leaky_relu")
    close(sol.leaky_relu_grad(X, 0.1), [0.1, 0.1, 0.1, 1, 1], "leaky_relu_grad")


def test_sigmoid():
    """sigmoid và sigmoid_grad"""
    close(sol.sigmoid(X), 1 / (1 + np.exp(-X)), "sigmoid")
    close(sol.sigmoid_grad(np.array([0.0])), [0.25], "sigmoid_grad(0)")
    close(sol.sigmoid_grad(XN), _num(sol.sigmoid, XN), "sigmoid_grad")


def test_tanh():
    """tanh và tanh_grad"""
    close(sol.tanh(X), np.tanh(X), "tanh")
    close(sol.tanh_grad(XN), _num(sol.tanh, XN), "tanh_grad")
