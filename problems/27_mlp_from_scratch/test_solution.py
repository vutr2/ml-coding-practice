import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def _params(seed=24):
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(size=(3, 5)) * 0.5, "b1": rng.normal(size=5) * 0.1,
            "W2": rng.normal(size=(5, 4)) * 0.5, "b2": rng.normal(size=4) * 0.1}


def test_gradient_check():
    """Gradient của cả 4 tham số khớp gradient số"""
    rng = np.random.default_rng(25)
    X, y = rng.normal(size=(8, 3)), rng.integers(0, 4, size=8)
    params = _params()
    out = sol.mlp_loss_and_grads(params, X, y)
    assert out is not None and len(out) == 2, "Phải trả về (loss, grads)"
    _, grads = out
    for k in ["W1", "b1", "W2", "b2"]:
        def f(v, k=k):
            p = dict(params)
            p[k] = v
            return sol.mlp_loss_and_grads(p, X, y)[0]
        close(grads[k], numerical_grad(f, params[k]), f"grads['{k}']", atol=1e-5)


def test_hai_vong_tron():
    """Dữ liệu hai vòng tròn (không tách tuyến tính): accuracy >= 95%"""
    rng = np.random.default_rng(26)
    n = 200
    r = np.r_[rng.uniform(0, 1, n), rng.uniform(2, 3, n)]
    t = rng.uniform(0, 2 * np.pi, 2 * n)
    X = np.c_[r * np.cos(t), r * np.sin(t)]
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    params = sol.train_mlp(X, y, hidden=32, lr=0.5, epochs=1000, seed=0)
    acc = np.mean(np.asarray(sol.mlp_predict(params, X)) == y)
    assert acc >= 0.95, f"Accuracy {acc:.2%} < 95%"
