import numpy as np


class SGDMomentum:
    def __init__(self, lr, momentum=0.9):
        # Viết code của bạn ở đây
        raise NotImplementedError

    def step(self, params, grads):
        raise NotImplementedError


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        # Viết code của bạn ở đây
        raise NotImplementedError

    def step(self, params, grads):
        raise NotImplementedError
