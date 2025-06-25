import numpy as np
from warnings import catch_warnings


def normpdf(index: int, mean: float, standard_deviation: float) -> float:
    # Copied from Quan's implementation in \Multiple Traits Setup\
    squared_deviation = float(standard_deviation) ** 2
    num = np.exp(-(float(index)-float(mean))**2 / (2 * squared_deviation))
    den = (2 * np.pi * squared_deviation) ** 0.5
    return num/den


def normal_state_priors(num_states: int) -> np.ndarray:
    # calculates state priors for a normal distrobution.
    mean = (num_states-1) / 2
    std_deviation = mean / 1.25

    state_probabilities = np.zeros((num_states,), dtype=np.float64)
    for i in range(num_states):
        state_probabilities[i] = normpdf(i, mean, std_deviation)  # matlab name

    state_probabilities /= np.sum(state_probabilities)
    return state_probabilities


def RaiseWarning(func):
    def wrapper(*args, **kwargs):
        with catch_warnings(action="error"):
            return func(*args, **kwargs)
    return wrapper


def transform(value):
    if value == 0:
        return 1.0
    elif value < 1:
        return 1.0/((value-1.0)**2)
    return (value+1.0)**2


def norm(arr):
    exp = np.exp(arr)
    exp_sum = np.sum(exp)
    return exp / exp_sum * 100


def stimgen(n: int) -> float:
    """Stimulus generalization function

    Args:
      n (int): the distance from the peak

    Returns:
      float: the coefficient of the contiguous reward
    """
    return 1 / 2**(n**2)
