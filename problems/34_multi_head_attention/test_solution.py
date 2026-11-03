import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _weights(d, seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4)]


def _attn(Q, K, V, mask=None):
    s = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])
    if mask is not None:
        s = np.where(mask, s, -1e9)
    e = np.exp(s - s.max(-1, keepdims=True))
    return (e / e.sum(-1, keepdims=True)) @ V


def test_mot_head():
    """num_heads = 1 bằng attention thường"""
    rng = np.random.default_rng(40)
    X = rng.normal(size=(5, 8))
    Wq, Wk, Wv, Wo = _weights(8, 41)
    exp = _attn(X @ Wq, X @ Wk, X @ Wv) @ Wo
    close(sol.multi_head_attention(X, Wq, Wk, Wv, Wo, 1), exp)


def test_nhieu_head():
    """num_heads = 4"""
    rng = np.random.default_rng(42)
    T, d, h = 6, 16, 4
    X = rng.normal(size=(T, d))
    Wq, Wk, Wv, Wo = _weights(d, 43)
    split = lambda M: M.reshape(T, h, d // h).transpose(1, 0, 2)
    out = _attn(split(X @ Wq), split(X @ Wk), split(X @ Wv))
    exp = out.transpose(1, 0, 2).reshape(T, d) @ Wo
    close(sol.multi_head_attention(X, Wq, Wk, Wv, Wo, h), exp)


def test_causal():
    """causal=True: đổi token sau không ảnh hưởng output token trước"""
    rng = np.random.default_rng(44)
    X = rng.normal(size=(5, 8))
    W = _weights(8, 45)
    out1 = np.asarray(sol.multi_head_attention(X, *W, 2, causal=True))
    X2 = X.copy()
    X2[3:] = rng.normal(size=(2, 8))
    out2 = np.asarray(sol.multi_head_attention(X2, *W, 2, causal=True))
    close(out1[:3], out2[:3], "output 3 token đầu")


def test_sai_so_head():
    """d_model không chia hết num_heads: ValueError"""
    X = np.zeros((3, 6))
    try:
        sol.multi_head_attention(X, *_weights(6, 46), 4)
    except ValueError:
        return
    raise AssertionError("Cần raise ValueError khi d_model không chia hết cho num_heads")


def test_so_voi_pytorch():
    """Khớp torch.nn.MultiheadAttention (bỏ qua nếu chưa cài torch)"""
    torch = _torch()
    if torch is None:
        return
    rng = np.random.default_rng(47)
    T, d, h = 5, 8, 2
    X = rng.normal(size=(T, d))
    Wq, Wk, Wv, Wo = _weights(d, 48)
    mha = torch.nn.MultiheadAttention(d, h, bias=False).double()
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.tensor(np.vstack([Wq.T, Wk.T, Wv.T])))
        mha.out_proj.weight.copy_(torch.tensor(Wo.T))
    block = ~torch.tril(torch.ones(T, T, dtype=torch.bool))   # PyTorch: True = bị chặn
    xt = torch.tensor(X)
    t, _ = mha(xt, xt, xt, attn_mask=block, need_weights=False)
    close(sol.multi_head_attention(X, Wq, Wk, Wv, Wo, h, causal=True), t.detach().numpy(), "so với PyTorch")
