"""Nạp file solution.py của từng bài cho test.

Mỗi file test gọi `sol = load(__file__)`. Bình thường hàm này nạp
`problems/<bài>/solution.py` (code của bạn). Khi đặt biến môi trường
ML_REF=1 (lệnh `python run.py --ref`), nó nạp lời giải mẫu trong `_solutions/`.
"""
import importlib.util
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(test_file):
    folder = os.path.dirname(os.path.abspath(test_file))
    name = os.path.basename(folder)
    use_ref = os.environ.get("ML_REF") == "1"
    if use_ref:
        path = os.path.join(ROOT, "_solutions", name + ".py")
    else:
        path = os.path.join(folder, "solution.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy {path}")
    spec = importlib.util.spec_from_file_location(
        f"solution_{name}_{'ref' if use_ref else 'user'}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(got, expected, what="Kết quả", atol=1e-6, rtol=1e-5):
    """So sánh số/mảng với đáp án, báo lỗi dễ đọc khi sai."""
    import numpy as np

    if got is None:
        raise AssertionError(f"{what}: hàm trả về None (quên `return`?)")
    try:
        got = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        raise AssertionError(f"{what}: kiểu trả về không hợp lệ ({type(got).__name__})")
    expected = np.asarray(expected, dtype=float)
    if got.shape != expected.shape:
        raise AssertionError(
            f"{what}: sai shape, nhận {got.shape}, cần {expected.shape}")
    if not np.allclose(got, expected, atol=atol, rtol=rtol, equal_nan=False):
        opts = dict(precision=4, threshold=24, edgeitems=3, suppress_small=True)
        raise AssertionError(
            f"{what}: sai giá trị\n"
            f"  nhận:   {np.array2string(got, **opts)}\n"
            f"  đáp án: {np.array2string(expected, **opts)}")


def numerical_grad(f, x, h=1e-6):
    """Gradient số (sai phân trung tâm), dùng trong test để kiểm tra backprop."""
    import numpy as np

    x = np.array(x, dtype=float)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + h
        fp = f(x)
        x[i] = old - h
        fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g
