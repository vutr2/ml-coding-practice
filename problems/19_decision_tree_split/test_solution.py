import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_gini():
    """Gini của vài mảng nhãn"""
    close(sol.gini(np.array([0, 0, 0])), 0.0, "gini([0,0,0])")
    close(sol.gini(np.array([0, 1, 0, 1])), 0.5, "gini([0,1,0,1])")
    close(sol.gini(np.array([0, 1, 2])), 2 / 3, "gini([0,1,2])")


def test_vi_du():
    """Ví dụ trong đề"""
    f, t, g = sol.best_split(np.array([[2.0], [3.0], [10.0], [19.0]]), np.array([0, 0, 1, 1]))
    assert f == 0, f"feature cần là 0, nhận {f}"
    close(t, 6.5, "threshold")
    close(g, 0.0, "weighted_gini")


def test_chon_cot_tot():
    """Cột 1 tách được hoàn toàn, cột 0 thì không"""
    X = np.array([[1, 5], [2, 1], [3, 6], [4, 2]], dtype=float)
    y = np.array([1, 0, 1, 0])
    f, t, g = sol.best_split(X, y)
    assert f == 1, f"feature cần là 1, nhận {f}"
    close(t, 3.5, "threshold")
    close(g, 0.0, "weighted_gini")


def test_hoa():
    """Hoà: chọn threshold nhỏ hơn"""
    f, t, g = sol.best_split(np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 1, 0, 1]))
    assert f == 0
    close(t, 1.5, "threshold")
    close(g, 1 / 3, "weighted_gini")


def test_khong_chia_duoc():
    """Mọi giá trị giống nhau: (None, None, gini(y))"""
    f, t, g = sol.best_split(np.array([[1.0, 2.0], [1.0, 2.0]]), np.array([0, 1]))
    assert f is None and t is None, f"Cần (None, None, ...), nhận ({f}, {t}, ...)"
    close(g, 0.5, "weighted_gini")
