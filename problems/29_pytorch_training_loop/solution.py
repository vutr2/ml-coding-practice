import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_model(in_dim, hidden, out_dim):
    # Viết code của bạn ở đây
    raise NotImplementedError


def train(model, X, y, epochs=30, lr=0.01, batch_size=32):
    """Trả về list loss trung bình của từng epoch."""
    # Viết code của bạn ở đây
    raise NotImplementedError


def evaluate(model, X, y):
    """Trả về accuracy (float)."""
    # Viết code của bạn ở đây
    raise NotImplementedError
