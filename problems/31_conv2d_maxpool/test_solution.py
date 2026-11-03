import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _conv_expected(x, w, b, s, p):
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    win = np.lib.stride_tricks.sliding_window_view(xp, w.shape[2:], axis=(1, 2))[:, ::s, ::s]
    return np.einsum("chwuv,ocuv->ohw", win, w) + b[:, None, None]


def test_vi_du():
    """Ví dụ conv2d trong đề"""
    x = np.arange(1.0, 10.0).reshape(1, 3, 3)
    w = np.array([[[[1.0, 0.0], [0.0, -1.0]]]])
    close(sol.conv2d(x, w, np.zeros(1)), [[[-4, -4], [-4, -4]]])


def test_padding():
    """padding = 1"""
    x = np.ones((1, 2, 2))
    w = np.ones((1, 1, 3, 3))
    close(sol.conv2d(x, w, np.array([0.5]), padding=1), [[[4.5, 4.5], [4.5, 4.5]]])


def test_nhieu_kenh_stride():
    """Nhiều kênh, stride 2, padding 1"""
    rng = np.random.default_rng(31)
    x, w, b = rng.normal(size=(3, 7, 7)), rng.normal(size=(4, 3, 3, 3)), rng.normal(size=4)
    close(sol.conv2d(x, w, b, stride=2, padding=1), _conv_expected(x, w, b, 2, 1))


def test_maxpool():
    """Ví dụ maxpool2d trong đề"""
    x = np.array([[[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14], [11, 12, 15, 16]]], dtype=float)
    close(sol.maxpool2d(x), [[[4, 8], [12, 16]]])


def test_so_voi_pytorch():
    """Khớp torch.nn.functional.conv2d (bỏ qua nếu chưa cài torch)"""
    torch = _torch()
    if torch is None:
        return
    rng = np.random.default_rng(32)
    x, w, b = rng.normal(size=(2, 6, 5)), rng.normal(size=(3, 2, 2, 3)), rng.normal(size=3)
    t = torch.nn.functional.conv2d(torch.tensor(x)[None], torch.tensor(w), torch.tensor(b), stride=1, padding=1)
    close(sol.conv2d(x, w, b, stride=1, padding=1), t[0].numpy())
