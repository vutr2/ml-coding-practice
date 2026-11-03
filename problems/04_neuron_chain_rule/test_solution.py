import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def _loss(x, w, b, y):
    a = 1 / (1 + np.exp(-(np.dot(w, x) + b)))
    return (a - y) ** 2


def _check(x, w, b, y):
    x, w = np.array(x, float), np.array(w, float)
    out = sol.neuron_forward_backward(x, w, b, y)
    assert out is not None and len(out) == 3, "Hàm phải trả về bộ ba (L, dw, db)"
    L, dw, db = out
    close(L, _loss(x, w, b, y), "L")
    close(dw, numerical_grad(lambda v: _loss(x, v, b, y), w), "dw")
    close(db, numerical_grad(lambda v: _loss(x, w, v[0], y), np.array([b]))[0], "db")


def test_vi_du():
    """Ví dụ trong đề"""
    _check([1, 2], [0.5, -0.25], 0.1, 1)


def test_nhan_0():
    """Nhãn y = 0"""
    _check([3, -1, 2], [0.2, 0.4, -0.1], -0.3, 0)


def test_ngau_nhien():
    """Vector 10 chiều ngẫu nhiên"""
    rng = np.random.default_rng(2)
    _check(rng.normal(size=10), rng.normal(size=10) * 0.3, 0.05, 1)
