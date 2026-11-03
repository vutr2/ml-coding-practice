import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


EXP = np.exp([1, 2, 3]) / np.exp([1, 2, 3]).sum()


def test_vi_du():
    """softmax([1, 2, 3])"""
    close(sol.softmax(np.array([1.0, 2.0, 3.0])), EXP)


def test_so_lon():
    """Logits rất lớn [1000, 1001, 1002] không ra nan"""
    close(sol.softmax(np.array([1000.0, 1001.0, 1002.0])), EXP)


def test_so_am_lon():
    """Logits rất âm [-1000, -999, -998]"""
    close(sol.softmax(np.array([-1000.0, -999.0, -998.0])), EXP)


def test_mang_2_chieu():
    """Mảng 2D: softmax theo từng hàng"""
    z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [5.0, 1.0, -2.0]])
    e = np.exp(z - z.max(axis=1, keepdims=True))
    close(sol.softmax(z), e / e.sum(axis=1, keepdims=True))
