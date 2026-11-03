import numpy as np
from harness import load, close, numerical_grad

sol = load(__file__)


def test_vi_du():
    """Ví dụ trong đề"""
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
    C, labels = sol.kmeans(X, np.array([[0.0, 0.0], [10.0, 10.0]]))
    close(C, [[0, 0.5], [10, 10.5]], "centroids")
    close(labels, [0, 0, 1, 1], "labels")


def test_khoi_tao_xau():
    """Khởi tạo xấu vẫn hội tụ đúng"""
    rng = np.random.default_rng(20)
    A = rng.normal(size=(50, 2))
    B = rng.normal(size=(50, 2)) + 10
    X = np.vstack([A, B])
    C, labels = sol.kmeans(X, X[[0, 1]].copy())
    C = np.asarray(C)
    order = np.argsort(C[:, 0])
    close(C[order], [A.mean(0), B.mean(0)], "centroids")


def test_cum_rong():
    """Cụm rỗng giữ nguyên tâm cũ"""
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
    init = np.array([[0.0, 0.0], [10.0, 10.0], [100.0, 100.0]])
    C, labels = sol.kmeans(X, init)
    close(C, [[0, 0.5], [10, 10.5], [100, 100]], "centroids")
    close(init, [[0, 0], [10, 10], [100, 100]], "init_centroids sau khi gọi (không được sửa)")
