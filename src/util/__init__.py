import os
import csv
import numpy as np
from warnings import catch_warnings
from builtins import open


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


def linear_reward_fn(param: tuple[float, float], null_signal=False):
    def get_reward(state, action):
        if null_signal and action == -1:
            return 0
        return param[0] - param[1] * abs(state - action)

    return get_reward


def generalized_stimgen(arr: np.ndarray, y: int, x: int, reward: float):
    # Mutates arr in place with a generalized stimgen for some reward in row y.
    # Two pointers ;)
    left_pointer = right_pointer = x
    x_lim = arr.shape[1]

    for i in range(1, 4):  # arbitrary range. stimgen(0) = 1
        # calculate this step's reduced reward with the stimgen function
        reduced_reward = stimgen(i) * reward

        right_pointer += 1
        if right_pointer < x_lim:
            # give reduced reward to ith right neighbor if in game bounds
            arr[y, right_pointer] += reduced_reward

        left_pointer -= 1
        if left_pointer >= 0:
            # give reduced reward to the ith left neighbor if in game bounds
            arr[y, left_pointer] += reduced_reward

def load_weights(filename: str):
    sg_wts = []
    rc_wts = []
    payoff = 0
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        flag = True
        for row in reader:
            if flag:
                flag = False
            else:
                sg_wts.append([float(x) for x in row[4:7]])
                rc_wts.append([float(x) for x in row[7:9]])
                if len(row) == 10:
                    payoff = float(row[9])
    return sg_wts, rc_wts, payoff


def get_stats_by_folder(folder_name: str, success_threshold: float) -> dict:
    files = os.listdir(folder_name)
    files = [x for x in files if x[-3:] == "csv"]
    
    payoff_average = 0
    success_count = 0
    for fi in files:
        _, _, payoff = load_weights(folder_name + fi)
        payoff_average += payoff

        if payoff >= success_threshold:
            success_count += 1

    stats = {
        "success_count": success_count,
        "payoff_average": payoff_average,
    }

    return stats
