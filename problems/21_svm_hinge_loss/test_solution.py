import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_vi_du():
    """Ví dụ trong đề"""
    X = np.array([[2.0, 0.0], [-2.0, 0.0]])
    loss, dw, db = sol.svm_loss_and_grad(X, np.array([1.0, -1.0]), np.zeros(2), 0.0, 1.0)
    close(loss, 1.0, "loss")
    close(dw, [-2.0, 0.0], "dw")
    close(db, 0.0, "db")


def test_gradient_check():
    """So gradient với gradient số (xa điểm gãy)"""
    rng = np.random.default_rng(17)
    X = rng.normal(size=(25, 3))
    y = np.where(rng.random(25) > 0.5, 1.0, -1.0)
    w, b = np.array([0.3, -0.7, 0.2]), 0.1
    f = lambda v: sol.svm_loss_and_grad(X, y, v, b, 2.0)[0]
    close(sol.svm_loss_and_grad(X, y, w, b, 2.0)[1], numerical_grad(f, w), "dw")
    g = lambda v: sol.svm_loss_and_grad(X, y, w, v[0], 2.0)[0]
    close(sol.svm_loss_and_grad(X, y, w, b, 2.0)[2], numerical_grad(g, np.array([b]))[0], "db")


def test_huan_luyen():
    """Hai cụm tách được: accuracy >= 95%"""
    rng = np.random.default_rng(18)
    X = np.vstack([rng.normal(-2, 1, size=(80, 2)), rng.normal(2, 1, size=(80, 2))])
    y = np.r_[-np.ones(80), np.ones(80)]
    w, b = sol.train_svm(X, y, C=1.0, lr=0.01, epochs=1000)
    acc = np.mean(np.asarray(sol.svm_predict(X, w, b)) == y)
    assert acc >= 0.95, f"Accuracy {acc:.2%} < 95%"
