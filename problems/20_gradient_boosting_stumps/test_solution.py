import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_stump():
    """Ví dụ fit_stump trong đề"""
    t, lv, rv = sol.fit_stump(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 5.0, 5.0]))
    close([t, lv, rv], [2.5, 1.0, 5.0], "(threshold, left, right)")


def test_predict_stump():
    """predict_stump"""
    close(sol.predict_stump((2.5, -1.0, 3.0), np.array([0.0, 2.5, 2.6, 10.0])), [-1, -1, 3, 3])


def test_boosting_mot_buoc():
    """Ví dụ boosting trong đề"""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 1.0, 5.0, 5.0])
    close(sol.gradient_boosting_fit_predict(x, y, x, 1, 0.5), [2, 2, 4, 4])


def test_loss_giam():
    """Thêm cây thì MSE trên train giảm"""
    rng = np.random.default_rng(16)
    x = np.sort(rng.uniform(0, 6, 80))
    y = np.sin(x) + rng.normal(scale=0.1, size=80)
    mse = [np.mean((np.asarray(sol.gradient_boosting_fit_predict(x, y, x, n, 0.3)) - y) ** 2)
           for n in [0, 5, 50]]
    close(mse[0], np.var(y), "MSE khi n_estimators = 0")
    assert mse[0] > mse[1] > mse[2], f"MSE phải giảm dần, nhận {np.round(mse, 4)}"
    assert mse[2] < 0.05, f"MSE sau 50 cây quá cao: {mse[2]:.4f}"
