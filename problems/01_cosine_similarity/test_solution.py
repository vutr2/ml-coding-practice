import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_vi_du():
    """Ví dụ [1,2,3] và [4,5,6]"""
    close(sol.cosine_similarity([1, 2, 3], [4, 5, 6]), 32 / np.sqrt(14 * 77))


def test_cung_chieu_nguoc_chieu():
    """Cùng chiều = 1, ngược chiều = -1"""
    close(sol.cosine_similarity([1, 2], [3, 6]), 1.0, "Cùng chiều")
    close(sol.cosine_similarity([1, 2], [-2, -4]), -1.0, "Ngược chiều")


def test_vuong_goc():
    """Hai vector vuông góc"""
    close(sol.cosine_similarity([1, 0, 0], [0, 5, 0]), 0.0)


def test_vector_khong():
    """Vector 0 trả về 0.0"""
    close(sol.cosine_similarity([0, 0, 0], [1, 2, 3]), 0.0)


def test_ngau_nhien():
    """Vector ngẫu nhiên 128 chiều"""
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=128), rng.normal(size=128)
    close(sol.cosine_similarity(a, b), a @ b / np.sqrt((a @ a) * (b @ b)))
