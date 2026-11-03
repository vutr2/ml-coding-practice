import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_tong_binh_phuong():
    """f = x0² + x1² tại [3, 4]"""
    close(sol.numerical_gradient(lambda x: np.sum(x ** 2), np.array([3.0, 4.0])), [6, 8])


def test_ham_hon_hop():
    """f = x0·x1 + sin(x2)"""
    f = lambda x: x[0] * x[1] + np.sin(x[2])
    x = np.array([2.0, -1.0, 0.5])
    close(sol.numerical_gradient(f, x), [-1.0, 2.0, np.cos(0.5)])


def test_ma_tran():
    """x là ma trận: f = sum(W³) -> 3W²"""
    W = np.array([[1.0, -2.0], [0.5, 3.0]])
    close(sol.numerical_gradient(lambda W: np.sum(W ** 3), W), 3 * W ** 2, atol=1e-4)


def test_khong_doi_x():
    """Không làm thay đổi x đầu vào"""
    x = np.array([1.0, 2.0, 3.0])
    sol.numerical_gradient(lambda v: np.sum(v ** 2), x)
    close(x, [1.0, 2.0, 3.0], "x sau khi gọi hàm")
