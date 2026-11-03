import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _sig(z):
    return 1 / (1 + np.exp(-z))


def test_trong_so_0():
    """W = U = b = 0: i = f = o = 0.5, g = 0"""
    c_prev = np.array([[2.0, -1.0]])
    h, c = sol.lstm_cell(np.ones((1, 3)), np.zeros((1, 2)), c_prev,
                         np.zeros((3, 8)), np.zeros((2, 8)), np.zeros(8))
    close(c, 0.5 * c_prev, "c")
    close(h, 0.5 * np.tanh(0.5 * c_prev), "h")


def test_cong_quen():
    """f ≈ 1, i ≈ 0: trí nhớ được giữ nguyên"""
    H = 2
    b = np.zeros(4 * H)
    b[0:H] = -100     # i ≈ 0
    b[H:2 * H] = 100  # f ≈ 1
    b[3 * H:] = 100   # o ≈ 1
    c_prev = np.array([[0.3, -0.7]])
    rng = np.random.default_rng(33)
    h, c = sol.lstm_cell(rng.normal(size=(1, 3)), np.zeros((1, H)), c_prev,
                         rng.normal(size=(3, 8)) * 0.1, rng.normal(size=(2, 8)) * 0.1, b)
    close(c, c_prev, "c")
    close(h, np.tanh(c_prev), "h")


def test_chuoi():
    """lstm_forward: shape và bước cuối khớp lstm_cell"""
    rng = np.random.default_rng(34)
    T, N, D, H = 5, 2, 3, 4
    xs = rng.normal(size=(T, N, D))
    W, U, b = rng.normal(size=(D, 4 * H)) * 0.5, rng.normal(size=(H, 4 * H)) * 0.5, rng.normal(size=4 * H) * 0.1
    hs, cT = sol.lstm_forward(xs, np.zeros((N, H)), np.zeros((N, H)), W, U, b)
    assert np.shape(hs) == (T, N, H), f"hs sai shape: {np.shape(hs)}"
    h, c = np.zeros((N, H)), np.zeros((N, H))
    for t in range(T):
        z = xs[t] @ W + h @ U + b
        i, f, g, o = _sig(z[:, :H]), _sig(z[:, H:2*H]), np.tanh(z[:, 2*H:3*H]), _sig(z[:, 3*H:])
        c = f * c + i * g
        h = o * np.tanh(c)
    close(hs[-1], h, "hs[-1]")
    close(cT, c, "c_T")


def test_so_voi_pytorch():
    """Khớp torch.nn.LSTMCell (bỏ qua nếu chưa cài torch)"""
    torch = _torch()
    if torch is None:
        return
    rng = np.random.default_rng(35)
    N, D, H = 3, 4, 5
    x, h0, c0 = rng.normal(size=(N, D)), rng.normal(size=(N, H)), rng.normal(size=(N, H))
    W, U, b = rng.normal(size=(D, 4 * H)), rng.normal(size=(H, 4 * H)), rng.normal(size=4 * H)
    cell = torch.nn.LSTMCell(D, H).double()
    with torch.no_grad():
        cell.weight_ih.copy_(torch.tensor(W.T))
        cell.weight_hh.copy_(torch.tensor(U.T))
        cell.bias_ih.copy_(torch.tensor(b))
        cell.bias_hh.zero_()
        th, tc = cell(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))
    h, c = sol.lstm_cell(x, h0, c0, W, U, b)
    close(h, th.numpy(), "h so với PyTorch")
    close(c, tc.numpy(), "c so với PyTorch")
