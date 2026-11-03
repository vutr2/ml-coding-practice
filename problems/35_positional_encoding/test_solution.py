import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_vi_du():
    """positional_encoding(2, 4)"""
    exp = [[0, 1, 0, 1], [np.sin(1), np.cos(1), np.sin(0.01), np.cos(0.01)]]
    close(sol.positional_encoding(2, 4), exp)


def test_shape_va_vi_tri_0():
    """Shape và hàng vị trí 0"""
    pe = np.asarray(sol.positional_encoding(50, 16))
    assert pe.shape == (50, 16), f"Sai shape: {pe.shape}"
    close(pe[0], np.tile([0.0, 1.0], 8), "PE[0]")


def test_khoang_cach_tuong_doi():
    """PE[p] · PE[p+k] không phụ thuộc p"""
    pe = np.asarray(sol.positional_encoding(100, 32))
    k = 5
    dots = [pe[p] @ pe[p + k] for p in [0, 10, 40, 90]]
    close(dots, np.full(4, dots[0]), "PE[p]·PE[p+5] với p = 0, 10, 40, 90")
