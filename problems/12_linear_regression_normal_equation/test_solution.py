import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_vi_du():
    """y = 2x + 1"""
    w, b = sol.fit_linear_regression(np.array([[1.0], [2.0], [3.0]]), np.array([3.0, 5.0, 7.0]))
    close(w, [2.0], "w")
    close(b, 1.0, "b")


def test_nhieu_dac_trung():
    """y = 2x1 - 3x2 + 0.5x3 + 5, không nhiễu"""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(50, 3))
    y = X @ np.array([2.0, -3.0, 0.5]) + 5
    w, b = sol.fit_linear_regression(X, y)
    close(w, [2.0, -3.0, 0.5], "w")
    close(b, 5.0, "b")


def test_co_nhieu():
    """Dữ liệu có nhiễu: khớp nghiệm bình phương tối thiểu"""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 4))
    y = X @ rng.normal(size=4) + 1.5 + rng.normal(scale=0.5, size=200)
    theta = np.linalg.lstsq(np.c_[X, np.ones(200)], y, rcond=None)[0]
    w, b = sol.fit_linear_regression(X, y)
    close(w, theta[:4], "w")
    close(b, theta[4], "b")
