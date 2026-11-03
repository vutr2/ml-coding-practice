import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_dropout_eval():
    """training=False trả về x không đổi"""
    x = np.arange(6.0).reshape(2, 3)
    close(sol.dropout(x, 0.5, training=False), x)


def test_dropout_ti_le():
    """Tỉ lệ phần tử bị tắt xấp xỉ p, phần còn lại nhân 1/(1-p)"""
    x = np.full((400, 250), 2.0)
    out = np.asarray(sol.dropout(x, 0.3, training=True, rng=np.random.default_rng(28)))
    frac = np.mean(out == 0)
    assert abs(frac - 0.3) < 0.01, f"Tỉ lệ bị tắt {frac:.3f}, cần xấp xỉ 0.3"
    kept = out[out != 0]
    close(kept[:5], np.full(5, 2.0 / 0.7), "Giá trị phần tử giữ lại")
    assert abs(out.mean() - 2.0) < 0.05, f"Kỳ vọng phải giữ nguyên ≈ 2.0, nhận {out.mean():.3f}"


def test_batchnorm():
    """BatchNorm đúng công thức"""
    rng = np.random.default_rng(29)
    x = rng.normal(3, 2, size=(16, 4))
    gamma, beta = np.array([1.0, 2.0, 0.5, 1.5]), np.array([0.0, -1.0, 2.0, 0.5])
    xhat = (x - x.mean(0)) / np.sqrt(x.var(0) + 1e-5)
    close(sol.batchnorm_forward(x, gamma, beta), gamma * xhat + beta)


def test_batchnorm_thong_ke():
    """Sau BatchNorm: mean mỗi cột = beta"""
    rng = np.random.default_rng(30)
    x = rng.normal(-5, 10, size=(64, 3))
    out = np.asarray(sol.batchnorm_forward(x, np.ones(3), np.array([1.0, 2.0, 3.0])))
    close(out.mean(0), [1.0, 2.0, 3.0], "mean mỗi cột", atol=1e-6)
