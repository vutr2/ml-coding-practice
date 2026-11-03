import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_vi_du():
    """Ví dụ trong đề"""
    close(sol.one_hot([0, 2, 1, 2]), [[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])


def test_num_classes():
    """Chỉ định num_classes lớn hơn max + 1"""
    close(sol.one_hot(np.array([1, 0]), num_classes=4), [[0, 1, 0, 0], [1, 0, 0, 0]])


def test_nhan_ngoai_khoang():
    """Nhãn >= num_classes phải raise ValueError"""
    try:
        sol.one_hot([0, 3], num_classes=3)
    except ValueError:
        return
    raise AssertionError("Cần raise ValueError khi nhãn >= num_classes")


def test_nhan_am():
    """Nhãn âm phải raise ValueError"""
    try:
        sol.one_hot([0, -1, 1])
    except ValueError:
        return
    raise AssertionError("Cần raise ValueError khi có nhãn âm")
