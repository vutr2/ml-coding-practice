import numpy as np


def svm_loss_and_grad(X, y, w, b, C):
    """Trả về (loss, dw, db) cho hinge loss + L2."""
    # Viết code của bạn ở đây
    raise NotImplementedError


def train_svm(X, y, C=1.0, lr=0.01, epochs=1000):
    """Khởi tạo w = 0, b = 0, gradient descent. Trả về (w, b)."""
    # Viết code của bạn ở đây
    raise NotImplementedError


def svm_predict(X, w, b):
    """Trả về nhãn -1/+1."""
    # Viết code của bạn ở đây
    raise NotImplementedError
