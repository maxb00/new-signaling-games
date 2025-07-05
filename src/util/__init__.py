import os
import csv
import numpy as np
from math import inf
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


def delta_reward_fn(param: tuple[float, float], null_signal=False):
    def get_reward(state, action):
        if null_signal and action == -1:
            return 0
        return 1 if abs(state - action) == 0 else 0
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


def load_weights(filename: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    sg_wts = []
    rc_wts = []
    final_payoff = 0
    rolling_payoff_average = 0
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        flag = True
        headers: list[str] = []
        for row in reader:
            if flag:  # Skip the header row
                headers += row
                flag = False
                continue

            state_signal_weights = []
            signal_action_weights = []
            for i in range(len(row)):
                if headers[i].startswith("sn_sg"):
                    state_signal_weights.append(float(row[i]))
                elif headers[i].startswith("rc_ac"):
                    signal_action_weights.append(float(row[i]))
                elif headers[i] == "payoff":
                    final_payoff = float(row[i])
                elif headers[i] == "average_payoff":
                    rolling_payoff_average = float(row[i])

            sg_wts.append(state_signal_weights)
            rc_wts.append(signal_action_weights)

    return np.array(sg_wts), np.array(rc_wts), (final_payoff, rolling_payoff_average)


def get_stats_by_folder(folder_name: str, success_threshold: float, n_signals: int) -> dict:
    files = os.listdir(folder_name)
    files = [x for x in files if x[-3:] == "csv"]

    final_payoff_average = 0
    rolling_payoff_average = 0
    success_count = 0
    pooling_count = 0
    payoff_range = [inf, -inf]
    for fi in files:
        # load weights + results
        w_sender, w_receiver, payoff = load_weights(folder_name + fi)

        # update payoff total
        final_payoff_average += payoff[0]
        rolling_payoff_average += payoff[1]

        # update success count
        if rolling_payoff_average >= success_threshold:
            success_count += 1

        # update payoff range
        payoff_range = min(payoff, payoff_range[0]), max(
            payoff, payoff_range[1])

        # did this game include any unused signals?
        signals_used = set()
        for i in range(w_sender.shape[0]):
            state_signal = np.argmax(w_sender[i])
            signals_used.add(state_signal)
        if signals_used != set(range(n_signals)):
            # if there was an unused signal, this game is in a pooling equilibrium
            pooling_count += 1

    final_payoff_average /= len(files)
    rolling_payoff_average /= len(files)

    stats = {
        "success_count": success_count,
        "final_payoff_average": final_payoff_average,
        "rolling_payoff_average": rolling_payoff_average,
        "final_payoff_range": payoff_range,
        "pooling_count": pooling_count
    }

    return stats
