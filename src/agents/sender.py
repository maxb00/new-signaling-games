from collections.abc import Callable
import numpy as np
from ..util import RaiseWarning, stimgen
import pdb


class Sender:
    def __init__(self, n_state: int, n_signals: int, null_signal: bool,
                 stimulus_generalization: bool, transform_func: Callable | None) -> None:
        # constants
        self.num_states = n_state  # during signal choice, null will be n.
        self.num_signals = n_signals + (1 if null_signal else 0)
        self.null_used = null_signal
        self.stimgen = stimulus_generalization
        self.random = np.random.default_rng()
        self.transform_func = transform_func

        # the up-to-date weights
        self.signal_weights = np.ones((self.num_signals, self.num_states))

        # for saving weight history
        self.history = []

    # given a state, gen a signal
    @RaiseWarning
    def generate_signal(self, state: int, record=False) -> int:
        # Convert weights to probabilities
        try:
            # transform weights
            # nota bene: we often use some kind of reciprocal function/quadratic to smooth negative rewards
            if self.transform_func is not None:
                # converts my piecewise transformation into a vector I use for ops
                transformation_vector = np.vectorize(
                    self.transform_func, otypes=[float])
                # transform signal weights using piecewise. projects weights for sustainable growth
                transformed_weights = transformation_vector(
                    self.signal_weights)
                # sum the weight of all signals for each state
                col_sums = np.sum(transformed_weights, axis=0)
                # convert the transformed weights to probabilities of signal given state
                prob = transformed_weights / col_sums
            else:
                # sum the weight of all signals for each state
                col_sums = np.sum(self.signal_weights, axis=0)
                # convert the transformed weights to probabilities of signal given state
                prob = self.signal_weights / col_sums
        except RuntimeWarning:
            print("Sender failed to convert weights")
            pdb.set_trace()

        # choose signal, including the null signal
        try:
            # Choose a signal based on all signal probs for a given state
            signal = self.random.choice(self.num_signals, p=prob.T[state])
        except ValueError:
            print("Passed a bad value to prob")
            pdb.set_trace()

        if self.null_used and signal == self.num_signals-1:
            # for our purposes moving forward, we'd prefer signal be -1 rather than n.
            signal = -1

        # record for graphs
        if record:
            # we calculate all weights at once for better logkeeping.
            self.history.append(prob)

        return signal

    def update_signal_weights(self, state: int, signal: int, reward: float) -> None:
        # Update the number of "balls in the urn" for a state, signal pair.
        # Generalizes reward to nearby states if stimulus_generalization == true during setup.

        # Give direct reward
        self.signal_weights[signal, state] += reward

        # Stimulus generalization
        if self.stimgen:
            # Two pointers ;)
            left_pointer = right_pointer = state
            for i in range(1, 4):  # arbitrary range. stimgen(0) = 1
                # calculate this step's reduced reward with the stimgen function defined in /util
                reduced_reward = stimgen(i) * reward

                right_pointer += 1
                if right_pointer < self.num_states:
                    # give reduced reward to ith right neighbor if in game bounds
                    self.signal_weights[signal,
                                        right_pointer] += reduced_reward

                left_pointer -= 1
                if left_pointer >= 0:
                    # give reduced reward to the ith left neighbor if in game bounds
                    self.signal_weights[signal, left_pointer] += reduced_reward

    def print_signal_probs(self) -> None:
        # utility function for directly printing the probs from the most recent generate_signal()

        # get probs from history
        probs = self.history[-1]

        # Build Header  means: message | state
        print('m|s', end=' ')
        # add state numbers to header
        for i in range(self.num_states):
            print(f'{i:3}', end=' ')
        print()

        # print probs
        for signal in range(self.num_signals):
            # add signal number to front of row
            print(f'{signal:3}', end=' ')
            for state in range(self.num_states):
                # print probs :)
                print(f'{int(probs[signal, state]):3}', end=' ')
            print()
