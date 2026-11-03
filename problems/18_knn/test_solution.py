import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_vi_du():
    """Ví dụ trong đề"""
    Xtr = np.array([[0, 0], [1, 0], [5, 5], [6, 5]], dtype=float)
    out = sol.knn_predict(Xtr, np.array([0, 0, 1, 1]), np.array([[0.5, 0], [5.5, 5]]), 2)
    close(out, [0, 1])


def test_k_1():
    """k = 1 trên chính tập train cho lại đúng nhãn"""
    rng = np.random.default_rng(14)
    X = rng.normal(size=(30, 3))
    y = rng.integers(0, 3, size=30)
    close(sol.knn_predict(X, y, X, 1), y)


def test_hoa_phieu():
    """Hoà phiếu: chọn nhãn nhỏ nhất"""
    Xtr = np.array([[0.0], [1.0], [-1.0], [10.0]])
    out = sol.knn_predict(Xtr, np.array([2, 1, 3, 0]), np.array([[0.1]]), 2)
    close(out, [1])


def test_ba_lop():
    """3 cụm rõ ràng"""
    rng = np.random.default_rng(15)
    centers = np.array([[0, 0], [8, 0], [0, 8]])
    Xtr = np.vstack([c + rng.normal(size=(20, 2)) for c in centers])
    ytr = np.repeat([0, 1, 2], 20)
    Xte = centers + 0.2
    close(sol.knn_predict(Xtr, ytr, Xte, 5), [0, 1, 2])
