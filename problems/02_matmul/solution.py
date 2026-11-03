import numpy as np


def matmul(A, B):
    """Nhân ma trận A (m, n) với B (n, p). Shape không khớp -> ValueError."""
    # Viết code của bạn ở đây

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    m, n = A.shape
    n2, p = B.shape

    if n != n2:
        raise ValueError(f"Shape không khớp: A({m},{n}), B({n2},{p})")
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            C[i, j] = np.sum(A[i, :] * B[:, j])
    
    return C
            
