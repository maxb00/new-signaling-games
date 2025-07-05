from collections.abc import Callable
import numpy as np
from ..util import RaiseWarning, generalized_stimgen
import pdb


class Receiver:
    # The reciever that peeks!
    def __init__(self, n_actions: int, n_signals: int, n_states: int, rng: np.random.Generator, stimulus_generalization: bool, transform_func: Callable | None) -> None:
        # basic constants
        self.num_actions = n_actions
        self.num_signals = n_signals
        self.num_states = n_states
        self.stimgen = stimulus_generalization
        self.transform_func = transform_func

        # inherit game rng for reproducability
        self.random = rng

        # main storage structures for probs
        self.signal_action_weights = np.ones(
            (self.num_signals, self.num_actions))
        self.state_action_weights = np.ones(
            (self.num_states, self.num_actions))
        # TODO: Consider "meta-urn" scoring urn reliability

        # list of final signal_action_probs gen-by-gen
        self.signal_action_history = []
        self.state_action_history = []

        # structures for holding intermediate choices
        self.latest_state_action_choice = -1
        self.latest_signal_action_choice = -1

    @RaiseWarning  # throws an error when underflow warnings encountered
    def generate_action(self, signal: int, world_state: int, make_record: bool) -> int:
        # meaningful defualts for pylance
        signal_action_probs = np.zeros_like(self.signal_action_weights)
        state_action_probs = np.zeros_like(self.state_action_weights)
        action = -1

        try:
            # transform weights
            # nota bene: we often use some kind of reciprocal function/quadratic to smooth negative rewards
            if self.transform_func is not None:
                # first, transform the slice of signal-to-action weights
                transformation_vector = np.vectorize(
                    self.transform_func, otypes=[float])
                transformed_signal_action_weights: np.ndarray = transformation_vector(
                    self.signal_action_weights)
                # Each row represents a signal;
                # sum the transformed propensities to convert them into action proabilities given a signal
                row_sums = np.sum(transformed_signal_action_weights, axis=1)
                signal_action_probs: np.ndarray = transformed_signal_action_weights.T / row_sums

                # calculate state-action probs for recording or usage if we are observing the world
                # transform to workable weight scores
                transformed_state_action_weights: np.ndarray = transformation_vector(
                    self.state_action_weights)
                # sum transformed propensities to convert to action probabilities
                state_action_sum = np.sum(
                    transformed_state_action_weights, axis=1)
                state_action_probs = transformed_state_action_weights.T / state_action_sum

            else:
                row_sums = np.sum(self.signal_action_weights, axis=1)
                signal_action_probs: np.ndarray = self.signal_action_weights.T / row_sums

                state_action_sum = np.sum(self.state_action_weights, axis=1)
                state_action_probs = self.state_action_weights.T / state_action_sum

            # before pooling, pick an action from each "urn" to determine if that "urn" is rewarded
            self.latest_state_action_choice = self.random.choice(
                self.num_actions, p=state_action_probs.T[world_state])
            self.latest_signal_action_choice = self.random.choice(
                self.num_actions, p=signal_action_probs.T[signal])

            if world_state != -1:
                # we are observing the world state, and should consider additonal propensities
                # dump the buckets together (add pre-transformation weights) and recalculate true propensities
                # the score for each state will be added to the buckets for all signals.
                transformed_signal_action_weights: np.ndarray = transformation_vector(
                    self.signal_action_weights + self.state_action_weights[world_state])
                # sum the transformed propensities to convert them into action proabilities given a signal
                row_sums = np.sum(transformed_signal_action_weights, axis=1)
                signal_action_probs: np.ndarray = transformed_signal_action_weights.T / row_sums

        except RuntimeWarning:
            print("Reciever failed to covert weights")
            pdb.set_trace()

        # choose action
        try:
            if signal == -1:
                action = -1
            else:
                action = self.random.choice(
                    self.num_actions, p=signal_action_probs.T[signal])
        except ValueError as e:
            print("Passed a bad value to prob")
            pdb.set_trace()

        # record if needed
        if make_record:
            self.signal_action_history.append(signal_action_probs.T)
            self.state_action_history.append(state_action_probs.T)

        return action

    def update_action_weights(self, signal: int, action: int, reward: float, world_state: int) -> None:
        # Update the number of "balls in the urn" for a state, signal pair.
        # Generalizes reward to nearby states if stimulus_generalization == true during setup.

        # If we observed the world state, we're going to refer to our saved latest choices
        if world_state != -1:  # we peeked!
            signal_action_gets_reward = self.latest_signal_action_choice != -1 \
                and self.latest_signal_action_choice == action
            if signal_action_gets_reward:
                # we picked the correct ball from the signal-to-action urn, and we can give it a reward
                self.signal_action_weights[signal, action] += reward
                if self.stimgen:
                    generalized_stimgen(
                        self.signal_action_weights, signal, action, reward)

            state_action_gets_reward = self.latest_state_action_choice != -1 \
                and self.latest_state_action_choice == action
            if state_action_gets_reward:
                # we picked the correct ball from the state-to-action urn, and we can give it a reward
                self.state_action_weights[world_state, action] += reward
                if self.stimgen:
                    generalized_stimgen(
                        self.state_action_weights, world_state, action, reward)

        else:  # Give direct reward
            self.signal_action_weights[signal, action] += reward

            if self.stimgen:
                generalized_stimgen(
                    self.signal_action_weights, signal, action, reward)

    def print_action_probs(self) -> None:
        # utility function for directly printing the probs from the most recent generate_action()

        # get probs from history
        probs = self.signal_action_history[-1]

        # Build Header  means: message | state
        print('m|a', end=' ')
        # add state numbers to header
        for i in range(self.num_actions):
            print(f'{i:3}', end=' ')
        print()

        # print probs
        for signal in range(self.num_signals):
            # add signal number to front of row
            print(f'{signal:3}', end=' ')
            for action in range(self.num_actions):
                # print probs :)
                print(f'{int(probs[signal, action]):3}', end=' ')
            print()
