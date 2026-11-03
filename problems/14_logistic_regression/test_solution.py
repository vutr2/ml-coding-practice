import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


import warnings


def test_sigmoid():
    """sigmoid(0) = 0.5 và ổn định với ±1000"""
    close(sol.sigmoid(np.array([0.0])), [0.5], "sigmoid(0)")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            out = sol.sigmoid(np.array([-1000.0, 1000.0]))
        except RuntimeWarning as e:
            raise AssertionError(f"sigmoid bị tràn số: {e}")
    close(out, [0.0, 1.0], "sigmoid([-1000, 1000])")


def test_loss_vi_du():
    """Ví dụ trong đề"""
    loss, dw, db = sol.bce_loss_and_grad(np.array([[1.0], [-1.0]]), np.array([1.0, 0.0]), np.zeros(1), 0.0)
    close(loss, np.log(2), "loss")
    close(dw, [-0.5], "dw")
    close(db, 0.0, "db")


def test_gradient_check():
    """So gradient với gradient số"""
    rng = np.random.default_rng(10)
    X, y = rng.normal(size=(40, 3)), (rng.random(40) > 0.5).astype(float)
    w, b = rng.normal(size=3) * 0.5, -0.2
    f = lambda v: sol.bce_loss_and_grad(X, y, v, b)[0]
    close(sol.bce_loss_and_grad(X, y, w, b)[1], numerical_grad(f, w), "dw")


def test_huan_luyen():
    """Huấn luyện trên 2 cụm điểm: accuracy >= 95%"""
    rng = np.random.default_rng(11)
    X = np.vstack([rng.normal(-2, 1, size=(100, 2)), rng.normal(2, 1, size=(100, 2))])
    y = np.r_[np.zeros(100), np.ones(100)]
    w, b = sol.train_logistic_regression(X, y, lr=0.1, epochs=500)
    pred = np.asarray(sol.predict(X, w, b))
    acc = np.mean(pred == y)
    assert acc >= 0.95, f"Accuracy {acc:.2%} < 95%"
