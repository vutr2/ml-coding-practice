import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_vi_du():
    """Ví dụ trong đề"""
    close(sol.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)


def test_hoan_hao_va_nguoc():
    """Xếp hạng hoàn hảo = 1, ngược hoàn toàn = 0"""
    close(sol.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0, "hoàn hảo")
    close(sol.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0, "ngược")


def test_hoa_diem():
    """Mọi điểm bằng nhau = 0.5"""
    close(sol.roc_auc([0, 1, 0, 1, 1], [0.5] * 5), 0.5)


def test_ngau_nhien():
    """Dữ liệu ngẫu nhiên có điểm trùng"""
    rng = np.random.default_rng(12)
    y = rng.integers(0, 2, size=60)
    s = np.round(rng.random(60) + 0.3 * y, 1)
    pos, neg = s[y == 1], s[y == 0]
    expected = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
    close(sol.roc_auc(y, s), expected)


def test_mot_lop():
    """Chỉ có một lớp phải raise ValueError"""
    try:
        sol.roc_auc([1, 1, 1], [0.2, 0.5, 0.9])
    except ValueError:
        return
    raise AssertionError("Cần raise ValueError khi y_true chỉ có một lớp")
