import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


YT = np.array([1, 0, 1, 1, 0, 0, 1])
YP = np.array([1, 0, 0, 1, 1, 0, 1])


def test_confusion_matrix():
    """Ví dụ trong đề"""
    close(sol.confusion_matrix(YT, YP), [[2, 1], [1, 3]])


def test_precision_recall_f1():
    """Ví dụ trong đề"""
    p, r, f = sol.precision_recall_f1(YT, YP)
    close(p, 0.75, "precision")
    close(r, 0.75, "recall")
    close(f, 0.75, "f1")


def test_lech():
    """Precision khác recall"""
    p, r, f = sol.precision_recall_f1([1, 1, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0])
    close(p, 1.0, "precision")
    close(r, 0.25, "recall")
    close(f, 0.4, "f1")


def test_chia_cho_0():
    """Không dự đoán lớp 1 nào: precision = 0.0"""
    p, r, f = sol.precision_recall_f1([1, 0, 1], [0, 0, 0])
    close([p, r, f], [0.0, 0.0, 0.0], "(precision, recall, f1)")
