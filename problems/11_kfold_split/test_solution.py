import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def _check_partition(folds, n, k):
    assert folds is not None and len(folds) == k, f"Cần {k} fold"
    all_val = []
    for i, (tr, va) in enumerate(folds):
        tr, va = set(np.asarray(tr).tolist()), set(np.asarray(va).tolist())
        assert not (tr & va), f"Fold {i}: train và val bị trùng chỉ số {sorted(tr & va)[:5]}"
        assert tr | va == set(range(n)), f"Fold {i}: train + val phải đủ 0..{n-1}"
        all_val += list(va)
    assert sorted(all_val) == list(range(n)), "Mỗi chỉ số phải nằm trong val đúng một lần"


def test_vi_du():
    """kfold_indices(10, 3) không shuffle"""
    folds = sol.kfold_indices(10, 3)
    _check_partition(folds, 10, 3)
    expected = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    for i, (tr, va) in enumerate(folds):
        assert list(np.asarray(va)) == expected[i], f"Fold {i}: val cần {expected[i]}, nhận {list(va)}"


def test_chia_deu():
    """n chia hết cho k"""
    folds = sol.kfold_indices(12, 4)
    _check_partition(folds, 12, 4)
    assert [len(va) for _, va in folds] == [3, 3, 3, 3]


def test_shuffle():
    """shuffle=True: vẫn chia đúng, cố định theo seed"""
    f1 = sol.kfold_indices(20, 5, shuffle=True, seed=7)
    f2 = sol.kfold_indices(20, 5, shuffle=True, seed=7)
    _check_partition(f1, 20, 5)
    perm = np.random.default_rng(7).permutation(20)
    assert list(np.asarray(f1[0][1])) == list(perm[:4]), "Fold 0 cần là 4 phần tử đầu của permutation(20)"
    for (a, b), (c, d) in zip(f1, f2):
        assert list(a) == list(c) and list(b) == list(d), "Cùng seed phải ra cùng kết quả"
