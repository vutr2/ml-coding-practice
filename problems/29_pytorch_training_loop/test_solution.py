import numpy as np
import torch
from torch import nn
from harness import load, close

sol = load(__file__)


def _data():
    g = torch.Generator().manual_seed(0)
    centers = torch.tensor([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    X = torch.cat([c + torch.randn(100, 2, generator=g) for c in centers])
    y = torch.arange(3).repeat_interleave(100)
    return X, y


def test_build_model():
    """Kiến trúc Linear → ReLU → Linear"""
    m = sol.build_model(2, 16, 3)
    assert isinstance(m, nn.Module), "build_model phải trả về nn.Module"
    linears = [l for l in m.modules() if isinstance(l, nn.Linear)]
    assert len(linears) == 2, f"Cần 2 lớp Linear, có {len(linears)}"
    assert linears[0].weight.shape == (16, 2) and linears[1].weight.shape == (3, 16), "Sai kích thước lớp Linear"
    assert m(torch.zeros(5, 2)).shape == (5, 3), "Đầu ra phải có shape (batch, out_dim)"


def test_train_loss_giam():
    """train trả về loss mỗi epoch, loss giảm"""
    torch.manual_seed(0)
    X, y = _data()
    m = sol.build_model(2, 16, 3)
    hist = sol.train(m, X, y, epochs=20, lr=0.01, batch_size=32)
    assert hist is not None and len(hist) == 20, "Cần trả về list 20 giá trị loss"
    assert hist[-1] < 0.5 * hist[0], f"Loss chưa giảm đủ: đầu {hist[0]:.3f}, cuối {hist[-1]:.3f}"


def test_evaluate():
    """Accuracy sau huấn luyện >= 95%"""
    torch.manual_seed(1)
    X, y = _data()
    m = sol.build_model(2, 16, 3)
    sol.train(m, X, y, epochs=30, lr=0.01, batch_size=32)
    acc = sol.evaluate(m, X, y)
    assert isinstance(acc, float), f"evaluate phải trả về float, nhận {type(acc).__name__}"
    assert acc >= 0.95, f"Accuracy {acc:.2%} < 95%"
    assert not m.training, "evaluate cần gọi model.eval()"
