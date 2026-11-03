import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_vi_du():
    """Ví dụ trong đề"""
    loss, dw, db = sol.mse_loss_and_grad(np.array([[1.0], [2.0]]), np.array([2.0, 4.0]), np.zeros(1), 0.0)
    close(loss, 10.0, "loss")
    close(dw, [-10.0], "dw")
    close(db, -6.0, "db")


def test_gradient_check():
    """So gradient với gradient số"""
    rng = np.random.default_rng(8)
    X, y = rng.normal(size=(30, 3)), rng.normal(size=30)
    w, b = rng.normal(size=3), 0.3
    f = lambda v: sol.mse_loss_and_grad(X, y, v, b)[0]
    close(sol.mse_loss_and_grad(X, y, w, b)[1], numerical_grad(f, w), "dw")
    g = lambda v: sol.mse_loss_and_grad(X, y, w, v[0])[0]
    close(sol.mse_loss_and_grad(X, y, w, b)[2], numerical_grad(g, np.array([b]))[0], "db")


def test_huan_luyen():
    """Huấn luyện tìm lại w, b thật"""
    rng = np.random.default_rng(9)
    X = rng.normal(size=(200, 2))
    y = X @ np.array([3.0, -1.0]) + 2.0
    w, b = sol.train_linear_regression_gd(X, y, lr=0.1, epochs=2000)
    close(w, [3.0, -1.0], "w", atol=1e-3)
    close(b, 2.0, "b", atol=1e-3)
