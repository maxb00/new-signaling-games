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


def get_stats_by_folder(folder_name: str, success_threshold: float, n_signals: int, pooling_threshold: float, n_seeds: int) -> dict:
    files = os.listdir(folder_name)
    files = [x for x in files if x[-3:] == "csv"]

    final_payoff_average = 0
    rolling_payoff_average = 0
    success_count = 0
    pooling_count = 0
    final_payoff_range = [inf, -inf]
    rolling_payoff_range = [inf, -inf]
    seeds_with_pooling = []
    min_payoff_seed = max_payoff_seed = min_rolling_seed = max_rolling_seed = 0
    #            0 - 0.5 - 0.75 - 0.875 - 1  
    payoff_range_buckets = [[], [], [], []]
    for fi in files:
        # load weights + results
        w_sender, w_receiver, payoff = load_weights(folder_name + fi)
        final_payoff, rolling_payoff = payoff

        # determine seed (strip extension and grab last _ section)
        seed = int(fi[:-4].split('_')[-1])

        # update payoff total
        final_payoff_average += final_payoff
        rolling_payoff_average += rolling_payoff

        # update success count
        if payoff[1] >= success_threshold:
            success_count += 1

        # update payoff range, save seeds
        if final_payoff < final_payoff_range[0]:
            final_payoff_range[0] = final_payoff
            min_payoff_seed = seed

        if final_payoff > final_payoff_range[1]:
            final_payoff_range[1] = final_payoff
            max_payoff_seed = seed

        if rolling_payoff < rolling_payoff_range[0]:
            rolling_payoff_range[0] = rolling_payoff
            min_rolling_seed = seed

        if rolling_payoff > rolling_payoff_range[1]:
            rolling_payoff_range[1] = rolling_payoff
            max_rolling_seed = seed

        # drop seed into payoff bucket
        if rolling_payoff <= 0.5:
            payoff_range_buckets[0].append(seed)
        elif rolling_payoff <= 0.75:
            payoff_range_buckets[1].append(seed)
        elif rolling_payoff <= 0.875:
            payoff_range_buckets[2].append(seed)
        else:
            payoff_range_buckets[3].append(seed)

        # did this game include any unused signals?
        signals_used = set()
        for i in range(w_sender.shape[0]):
            try:
                # choose first signal above pooling_threshold
                state_signal = np.where(w_sender[i] > pooling_threshold)[0][0]
                signals_used.add(state_signal)
            except IndexError:
                # no signals were above threshold
                continue
        if signals_used != set(range(n_signals)):
            # if there was an unused signal, this game is in a pooling equilibrium
            pooling_count += 1
            seeds_with_pooling.append(seed)

    final_payoff_average /= len(files)
    rolling_payoff_average /= len(files)

    stats = {
        "success_count": success_count,
        "final_payoff_range": final_payoff_range,
        "final_payoff_average": final_payoff_average,
        "final_payoff_seeds": [min_payoff_seed, max_payoff_seed],
        "rolling_payoff_range": rolling_payoff_range,
        "rolling_payoff_average": rolling_payoff_average,
        "rolling_payoff_seeds": [min_rolling_seed, max_rolling_seed],
        "pooling_count": pooling_count,
        "pooling_seeds": seeds_with_pooling[:n_seeds],
        "<=0.5_count": len(payoff_range_buckets[0]),
        "<=0.5_seeds": payoff_range_buckets[0][:n_seeds],
        "0.5-0.75_count": len(payoff_range_buckets[1]),
        "0.5-0.75_seeds": payoff_range_buckets[1][:n_seeds],
        "0.75-0.875_count": len(payoff_range_buckets[2]),
        "0.75-0.875_seeds": payoff_range_buckets[2][:n_seeds],
        "0.875-1_count": len(payoff_range_buckets[3]),
        "0.875-1_seeds": payoff_range_buckets[3][:n_seeds]
    }

    return stats
