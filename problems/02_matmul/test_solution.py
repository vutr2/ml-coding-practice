import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_vi_du():
    """Ví dụ trong đề (2x3 nhân 3x2)"""
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8], [9, 10], [11, 12]]
    close(sol.matmul(A, B), [[58, 64], [139, 154]])


def test_ma_tran_don_vi():
    """Nhân với ma trận đơn vị giữ nguyên"""
    A = np.arange(12, dtype=float).reshape(3, 4)
    close(sol.matmul(A, np.eye(4)), A)


def test_vector_hang_cot():
    """(1, n) nhân (n, 1) ra (1, 1)"""
    close(sol.matmul([[1, 2, 3]], [[4], [5], [6]]), [[32]])


def test_ngau_nhien():
    """Ma trận ngẫu nhiên 5x7 nhân 7x4"""
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(5, 7)), rng.normal(size=(7, 4))
    close(sol.matmul(A, B), A @ B)


def test_sai_shape():
    """Shape không khớp phải raise ValueError"""
    try:
        sol.matmul(np.ones((2, 3)), np.ones((2, 3)))
    except ValueError:
        return
    raise AssertionError("Cần raise ValueError khi số cột của A khác số hàng của B")
