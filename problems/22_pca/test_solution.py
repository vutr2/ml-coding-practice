import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def _data():
    rng = np.random.default_rng(19)
    Z = rng.normal(size=(300, 3)) * np.array([5.0, 2.0, 0.5])
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return Z @ Q.T + np.array([1.0, -2.0, 3.0])


def _expected(X, k):
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k]
    idx = np.argmax(np.abs(comps), axis=1)
    comps = comps * np.sign(comps[np.arange(k), idx])[:, None]
    return comps, S[:k] ** 2 / (len(X) - 1), Xc @ comps.T


def test_shape():
    """Shape của 3 kết quả"""
    out = sol.pca(_data(), 2)
    assert out is not None and len(out) == 3, "Phải trả về (components, explained_variance, X_proj)"
    shapes = tuple(np.asarray(o).shape for o in out)
    assert shapes == ((2, 3), (2,), (300, 2)), f"Sai shape: {shapes}"


def test_gia_tri():
    """components, explained_variance, X_proj đúng (theo quy ước dấu)"""
    X = _data()
    comps, var, proj = sol.pca(X, 2)
    ec, ev, ep = _expected(X, 2)
    close(var, ev, "explained_variance")
    close(comps, ec, "components")
    close(proj, ep, "X_proj", atol=1e-5)


def test_truc_chuan():
    """Các thành phần chính trực chuẩn"""
    comps, _, _ = sol.pca(_data(), 3)
    close(np.asarray(comps) @ np.asarray(comps).T, np.eye(3), "components @ components.T")
