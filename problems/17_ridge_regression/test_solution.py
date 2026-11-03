import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def _data(seed=13):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 3)) + 2
    y = X @ np.array([1.5, -2.0, 0.7]) + 4 + rng.normal(scale=0.3, size=60)
    return X, y


def test_alpha_0():
    """alpha = 0 giống linear regression"""
    X, y = _data()
    theta = np.linalg.lstsq(np.c_[X, np.ones(len(y))], y, rcond=None)[0]
    w, b = sol.fit_ridge(X, y, 0.0)
    close(w, theta[:3], "w")
    close(b, theta[3], "b")


def test_alpha_10():
    """alpha = 10: đúng công thức nghiệm đóng"""
    X, y = _data()
    Xc, yc = X - X.mean(0), y - y.mean()
    w_exp = np.linalg.solve(Xc.T @ Xc + 10 * np.eye(3), Xc.T @ yc)
    w, b = sol.fit_ridge(X, y, 10.0)
    close(w, w_exp, "w")
    close(b, y.mean() - X.mean(0) @ w_exp, "b")


def test_alpha_rat_lon():
    """alpha rất lớn: w gần 0, b gần mean(y)"""
    X, y = _data()
    w, b = sol.fit_ridge(X, y, 1e9)
    close(w, np.zeros(3), "w", atol=1e-4)
    close(b, y.mean(), "b", atol=1e-3)


def test_w_nho_dan():
    """Tăng alpha thì ||w|| giảm"""
    X, y = _data()
    norms = [np.linalg.norm(sol.fit_ridge(X, y, a)[0]) for a in [0.1, 10, 100, 1000]]
    assert all(norms[i] > norms[i + 1] for i in range(3)), f"||w|| phải giảm dần, nhận {np.round(norms, 3)}"
