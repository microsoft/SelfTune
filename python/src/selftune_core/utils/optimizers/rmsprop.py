import numpy as np

__all__ = "RMSprop"


class RMSprop:
    def __init__(self, eta: float = 0.01, alpha: float = 0.99, momentum: float = 0.0, eps: float = 1e-8):
        self.eta = eta
        self.alpha = alpha
        self.momentum = momentum
        self.eps = eps

        # Initializations
        self.square_average = 0
        self.buffer = 0

    def get_step_value(self, grad: np.ndarray):
        """Implementation picked up from https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html."""
        self.square_average = self.alpha * self.square_average + (1 - self.alpha) * np.square(grad)
        step_value = None

        if self.momentum > 0:
            self.buffer = self.momentum * self.buffer + grad / (np.sqrt(self.square_average) + self.eps)
            step_value = self.eta * self.buffer
        else:
            step_value = self.eta * grad / (np.sqrt(self.square_average) + self.eps)
        return step_value
