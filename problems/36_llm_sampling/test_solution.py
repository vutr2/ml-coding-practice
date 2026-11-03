import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


L = np.log(np.array([0.5, 0.3, 0.15, 0.05]))


def _kept(x):
    return list(np.flatnonzero(np.isfinite(np.asarray(x))))


def test_top_k():
    """top_k_filter giữ k logits lớn nhất"""
    out = sol.top_k_filter(L, 2)
    assert _kept(out) == [0, 1], f"Cần giữ token [0, 1], nhận {_kept(out)}"
    close(np.asarray(out)[:2], L[:2], "Giá trị logits được giữ")
    z = np.array([0.1, 3.0, -1.0, 2.0, 0.5])
    assert _kept(sol.top_k_filter(z, 3)) == [1, 3, 4], "top_k_filter([0.1, 3, -1, 2, 0.5], 3) cần giữ [1, 3, 4]"


def test_top_p():
    """top_p_filter với p = 0.75, 0.4, 0.99"""
    assert _kept(sol.top_p_filter(L, 0.75)) == [0, 1], f"p = 0.75: cần [0, 1], nhận {_kept(sol.top_p_filter(L, 0.75))}"
    assert _kept(sol.top_p_filter(L, 0.4)) == [0], f"p = 0.4: cần [0], nhận {_kept(sol.top_p_filter(L, 0.4))}"
    assert _kept(sol.top_p_filter(L, 0.99)) == [0, 1, 2, 3], "p = 0.99: cần giữ cả 4 token"


def test_top_p_khong_sap_xep():
    """top_p_filter khi logits chưa sắp xếp"""
    z = np.log(np.array([0.1, 0.6, 0.05, 0.25]))
    assert _kept(sol.top_p_filter(z, 0.8)) == [1, 3], f"Cần [1, 3], nhận {_kept(sol.top_p_filter(z, 0.8))}"


def test_khong_sua_dau_vao():
    """Không sửa mảng logits đầu vào"""
    z = L.copy()
    sol.top_k_filter(z, 1)
    sol.top_p_filter(z, 0.4)
    close(z, L, "logits sau khi gọi hàm")


def test_greedy():
    """temperature = 0 chọn argmax"""
    assert int(sol.sample(np.array([0.1, 2.0, 1.9]), temperature=0)) == 1


def test_phan_phoi():
    """Tần suất lấy mẫu khớp softmax(logits / T)"""
    rng = np.random.default_rng(49)
    z = np.array([1.0, 2.0, 0.0])
    T = 0.7
    counts = np.bincount([int(sol.sample(z, T, rng)) for _ in range(6000)], minlength=3) / 6000
    p = np.exp(z / T) / np.exp(z / T).sum()
    assert np.abs(counts - p).max() < 0.03, f"Tần suất {np.round(counts, 3)}, cần gần {np.round(p, 3)}"


def test_ket_hop_filter():
    """Lấy mẫu sau top-k không bao giờ ra token bị lọc"""
    rng = np.random.default_rng(50)
    z = sol.top_k_filter(np.array([0.5, 1.0, 3.0, 2.5, 0.0]), 2)
    picks = {int(sol.sample(z, 1.0, rng)) for _ in range(300)}
    assert picks <= {2, 3}, f"Lấy mẫu ra token bị lọc: {sorted(picks - {2, 3})}"
