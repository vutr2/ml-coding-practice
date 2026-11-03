import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def _setup(seed=21):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(4, 3)), rng.normal(size=(3, 5)), rng.normal(size=5), rng.normal(size=(4, 5))


def test_forward():
    """linear_forward"""
    X, W, b, _ = _setup()
    close(sol.linear_forward(X, W, b), X @ W + b)


def test_backward_gradient_check():
    """dX, dW, db khớp gradient số (L = sum(Y * G))"""
    X, W, b, G = _setup()
    dX, dW, db = sol.linear_backward(X, W, G)
    L = lambda X_, W_, b_: np.sum((X_ @ W_ + b_) * G)
    close(dX, numerical_grad(lambda v: L(v, W, b), X), "dX")
    close(dW, numerical_grad(lambda v: L(X, v, b), W), "dW")
    close(db, numerical_grad(lambda v: L(X, W, v), b), "db")


def test_shape():
    """Shape gradient bằng shape biến"""
    X, W, b, G = _setup(22)
    dX, dW, db = sol.linear_backward(X, W, G)
    got = (np.shape(dX), np.shape(dW), np.shape(db))
    assert got == ((4, 3), (3, 5), (5,)), f"Sai shape (dX, dW, db): {got}"
