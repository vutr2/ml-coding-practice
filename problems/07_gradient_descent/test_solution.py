import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_hai_buoc_dau():
    """f = (x-3)²: hai bước đầu"""
    hist = sol.gradient_descent(lambda x: 2 * (x - 3), np.array([0.0]), 0.1, 2)
    close(hist, [[0.0], [0.6], [1.08]])


def test_hoi_tu():
    """Hội tụ về cực tiểu sau 200 bước"""
    hist = sol.gradient_descent(lambda x: 2 * (x - 3), np.array([0.0]), 0.1, 200)
    close(hist[-1], [3.0], "Điểm cuối", atol=1e-6)


def test_hai_chieu():
    """f = x² + 10y² (2 chiều)"""
    grad = lambda v: np.array([2 * v[0], 20 * v[1]])
    hist = sol.gradient_descent(grad, np.array([5.0, 1.0]), 0.05, 300)
    assert np.asarray(hist).shape == (301, 2), f"Sai shape lịch sử: {np.asarray(hist).shape}, cần (301, 2)"
    close(hist[1], [5.0 - 0.05 * 10, 1.0 - 0.05 * 20], "Bước 1")
    close(hist[-1], [0.0, 0.0], "Điểm cuối", atol=1e-6)


def test_khong_doi_x0():
    """Không làm thay đổi x0"""
    x0 = np.array([0.0])
    sol.gradient_descent(lambda x: 2 * (x - 3), x0, 0.1, 5)
    close(x0, [0.0], "x0 sau khi gọi hàm")
