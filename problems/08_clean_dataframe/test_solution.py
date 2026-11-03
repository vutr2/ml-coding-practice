import numpy as np
import pandas as pd
from harness import load, close

sol = load(__file__)


def _df():
    return pd.DataFrame({
        "age": [25.0, np.nan, 35.0, 25.0, 40.0],
        "salary": [10, 20, 30, 10, np.nan],
        "city": ["Hanoi", "HCM", None, "Hanoi", "Hanoi"],
    })


def test_xoa_trung():
    """Xoá hàng trùng và đánh lại index"""
    out = sol.clean(_df())
    assert out is not None, "Hàm trả về None (quên `return`?)"
    assert len(out) == 4, f"Cần 4 hàng sau khi xoá trùng, nhận {len(out)}"
    assert list(out.index) == [0, 1, 2, 3], f"Index cần là [0, 1, 2, 3], nhận {list(out.index)}"


def test_dien_cot_so():
    """Cột số điền bằng median (tính sau khi xoá trùng)"""
    out = sol.clean(_df())
    close(out["age"].to_numpy(), [25.0, 35.0, 35.0, 40.0], "Cột age")
    close(out["salary"].to_numpy(), [10.0, 20.0, 30.0, 20.0], "Cột salary")


def test_dien_cot_chu():
    """Cột chữ điền bằng "unknown" """
    out = sol.clean(_df())
    got = list(out["city"])
    assert got == ["Hanoi", "HCM", "unknown", "Hanoi"], f"Cột city sai: {got}"


def test_khong_sua_dau_vao():
    """Không thay đổi DataFrame đầu vào"""
    df = _df()
    sol.clean(df)
    assert len(df) == 5 and df["age"].isna().sum() == 1, "DataFrame đầu vào đã bị sửa"
