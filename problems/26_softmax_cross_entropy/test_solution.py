import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_vi_du():
    """Ví dụ trong đề"""
    z = np.array([[2.0, 1.0, 0.1]])
    p = np.exp(z) / np.exp(z).sum()
    loss, d = sol.softmax_cross_entropy(z, np.array([0]))
    close(loss, -np.log(p[0, 0]), "loss")
    close(d, p - np.array([[1, 0, 0]]), "dlogits")


def test_on_dinh():
    """Logits lớn (1000) không ra nan/inf"""
    z = np.array([[1000.0, 0.0], [0.0, 1000.0]])
    loss, d = sol.softmax_cross_entropy(z, np.array([1, 1]))
    close(loss, 500.0, "loss")
    close(d, [[0.5, -0.5], [0.0, 0.0]], "dlogits")


def test_gradient_check():
    """dlogits khớp gradient số"""
    rng = np.random.default_rng(23)
    z, y = rng.normal(size=(6, 4)), rng.integers(0, 4, size=6)
    _, d = sol.softmax_cross_entropy(z, y)
    close(d, numerical_grad(lambda v: sol.softmax_cross_entropy(v, y)[0], z), "dlogits")
