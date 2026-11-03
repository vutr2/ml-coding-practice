import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_vi_du():
    """Ví dụ trong đề, có cột hằng số"""
    X = np.array([[1.0, 10.0], [2.0, 10.0], [3.0, 10.0]])
    mean, std = sol.fit_standardizer(X)
    close(mean, [2.0, 10.0], "mean")
    close(std, [np.sqrt(2 / 3), 0.0], "std")
    z = np.sqrt(1.5)
    close(sol.transform(X, mean, std), [[-z, 0], [0, 0], [z, 0]], "transform")


def test_train_sau_chuan_hoa():
    """Tập train sau chuẩn hoá có mean 0, std 1"""
    rng = np.random.default_rng(4)
    X = rng.normal(5, 3, size=(100, 4))
    Z = sol.transform(X, *sol.fit_standardizer(X))
    close(np.asarray(Z).mean(axis=0), np.zeros(4), "mean sau chuẩn hoá", atol=1e-9)
    close(np.asarray(Z).std(axis=0), np.ones(4), "std sau chuẩn hoá")


def test_dung_thong_so_train_cho_test():
    """Tập test dùng mean/std của tập train"""
    X_train = np.array([[0.0], [2.0], [4.0]])
    X_test = np.array([[2.0], [6.0]])
    mean, std = sol.fit_standardizer(X_train)
    s = np.sqrt(8 / 3)
    close(sol.transform(X_test, mean, std), [[0.0], [4.0 / s]])
