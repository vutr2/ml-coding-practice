import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def test_causal_mask():
    """causal_mask(3)"""
    got = np.asarray(sol.causal_mask(3))
    exp = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=bool)
    assert got.dtype == bool and (got == exp).all(), f"causal_mask(3) sai:\n{got}"


def test_q_bang_0():
    """Q = 0: trọng số đều, output = trung bình V"""
    rng = np.random.default_rng(36)
    K, V = rng.normal(size=(4, 8)), rng.normal(size=(4, 3))
    out, w = sol.scaled_dot_product_attention(np.zeros((2, 8)), K, V)
    close(w, np.full((2, 4), 0.25), "weights")
    close(out, np.tile(V.mean(0), (2, 1)), "output")


def test_gia_tri():
    """Đúng công thức, có batch"""
    rng = np.random.default_rng(37)
    Q, K, V = rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 5, 4)), rng.normal(size=(2, 5, 6))
    s = Q @ np.swapaxes(K, -1, -2) / 2.0
    e = np.exp(s - s.max(-1, keepdims=True))
    w = e / e.sum(-1, keepdims=True)
    out, weights = sol.scaled_dot_product_attention(Q, K, V)
    close(weights, w, "weights")
    close(out, w @ V, "output")


def test_mask():
    """Với causal mask: trọng số phía trên đường chéo = 0, mỗi hàng tổng = 1"""
    rng = np.random.default_rng(38)
    X = rng.normal(size=(4, 8))
    _, w = sol.scaled_dot_product_attention(X, X, X, mask=np.tril(np.ones((4, 4), bool)))
    w = np.asarray(w)
    close(np.triu(w, 1), np.zeros((4, 4)), "weights phía trên đường chéo")
    close(w.sum(-1), np.ones(4), "tổng mỗi hàng")
    close(w[0, 0], 1.0, "token đầu chỉ nhìn chính nó")


def test_so_voi_pytorch():
    """Khớp F.scaled_dot_product_attention (bỏ qua nếu chưa cài torch)"""
    torch = _torch()
    if torch is None:
        return
    rng = np.random.default_rng(39)
    Q, K, V = rng.normal(size=(2, 5, 8)), rng.normal(size=(2, 5, 8)), rng.normal(size=(2, 5, 8))
    mask = np.tril(np.ones((5, 5), bool))
    t = torch.nn.functional.scaled_dot_product_attention(
        torch.tensor(Q), torch.tensor(K), torch.tensor(V), attn_mask=torch.tensor(mask))
    close(sol.scaled_dot_product_attention(Q, K, V, mask)[0], t.numpy(), "output so với PyTorch")
