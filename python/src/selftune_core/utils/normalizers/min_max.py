import numpy as np


def min_max_norm(values: np.ndarray, lbs: np.ndarray, ubs: np.ndarray) -> np.ndarray:
    return (values - lbs) / (ubs - lbs)


def min_max_denorm(normalized_values: np.ndarray, lbs: np.ndarray, ubs: np.ndarray) -> np.ndarray:
    return lbs + normalized_values * (ubs - lbs)
