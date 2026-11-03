import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def test_sgd_momentum():
    """SGD momentum: 3 bước với gradient bằng 1"""
    p = {"w": np.array([1.0])}
    opt = sol.SGDMomentum(lr=0.1, momentum=0.9)
    vals = []
    for _ in range(3):
        opt.step(p, {"w": np.array([1.0])})
        vals.append(p["w"][0])
    close(vals, [0.9, 0.71, 0.439], "θ sau mỗi bước")


def test_adam_buoc_dau():
    """Adam bước đầu: mọi tham số di chuyển đúng lr"""
    p = {"w": np.array([0.0, 0.0, 0.0])}
    sol.Adam(lr=0.1).step(p, {"w": np.array([0.01, -5.0, 100.0])})
    close(p["w"], [-0.1, 0.1, -0.1], "θ sau bước 1")


def test_nhieu_tham_so():
    """Trạng thái riêng cho từng tham số"""
    p = {"a": np.array([1.0]), "b": np.array([1.0, 2.0])}
    opt = sol.SGDMomentum(lr=0.1, momentum=0.5)
    opt.step(p, {"a": np.array([1.0]), "b": np.array([0.0, 2.0])})
    opt.step(p, {"a": np.array([1.0]), "b": np.array([0.0, 2.0])})
    close(p["a"], [1.0 - 0.1 - 0.15], "a")
    close(p["b"], [1.0, 2.0 - 0.2 - 0.3], "b")


def test_adam_hoi_tu():
    """Adam tối thiểu hoá (θ - 3)²"""
    p = {"w": np.array([0.0, 10.0])}
    opt = sol.Adam(lr=0.1)
    for _ in range(1000):
        opt.step(p, {"w": 2 * (p["w"] - 3)})
    close(p["w"], [3.0, 3.0], "θ cuối", atol=1e-2)


def test_so_voi_pytorch():
    """Khớp torch.optim.Adam (bỏ qua nếu chưa cài torch)"""
    torch = _torch()
    if torch is None:
        return
    rng = np.random.default_rng(27)
    w0 = rng.normal(size=4)
    gs = [rng.normal(size=4) for _ in range(5)]
    p = {"w": w0.copy()}
    opt = sol.Adam(lr=0.05)
    tw = torch.tensor(w0.copy(), requires_grad=True)
    topt = torch.optim.Adam([tw], lr=0.05)
    for g in gs:
        opt.step(p, {"w": g})
        tw.grad = torch.tensor(g)
        topt.step()
    close(p["w"], tw.detach().numpy(), "θ so với PyTorch")
