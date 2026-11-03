import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


X = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)


def test_log_likelihood():
    """Giá trị log-likelihood"""
    expected = np.sum(-0.5 * np.log(2 * np.pi * 2.5) - (X - 4.5) ** 2 / (2 * 2.5))
    close(sol.gaussian_log_likelihood(X, 4.5, 2.5), expected)


def test_mle_vi_du():
    """MLE của dữ liệu trong đề = (5, 4)"""
    out = sol.gaussian_mle(X)
    assert out is not None and len(out) == 2, "Phải trả về (mu, var)"
    close(out[0], 5.0, "mu")
    close(out[1], 4.0, "var")


def test_mle_la_cuc_dai():
    """Tham số MLE cho log-likelihood lớn nhất"""
    rng = np.random.default_rng(3)
    x = rng.normal(2.0, 1.5, size=200)
    mu, var = sol.gaussian_mle(x)
    best = sol.gaussian_log_likelihood(x, mu, var)
    for dm, dv in [(0.1, 0), (-0.1, 0), (0, 0.2), (0, -0.2)]:
        assert best >= sol.gaussian_log_likelihood(x, mu + dm, var + dv), \
            "Thay đổi mu/var làm log-likelihood tăng, vậy (mu, var) chưa phải MLE"
