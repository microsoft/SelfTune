import numpy as np

__all__ = "SGD"


class SGD:
    def __init__(self, eta: float):
        self.eta = eta

    def get_step_value(self, grad: np.ndarray):
        return self.eta * grad
