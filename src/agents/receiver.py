import numpy as np
from util import RaiseWarning, transform, stimgen
import pdb


class Receiver:
    # The reciever that peeks!
    def __init__(self, n_actions: int, n_signals: int, n_states: int, rng: np.random.Generator, stimulus_generalization: bool) -> None:
        self.num_actions = n_actions
        self.num_signals = n_signals
        self.num_states = n_states
        self.random = rng
        self.stimgen = stimulus_generalization

        self.signal_action_weights = np.ones(
            (self.num_signals, self.num_actions))
        self.state_action_weights = np.ones(
            (self.num_states, self.num_actions))

        self.history = []

    @RaiseWarning  # throws an error when underflow warnings encountered
    def generate_action(self, signal: int, world_state: int, make_record: bool) -> int:
        # transform weights
        try:
            # first, transform the slice of signal-to-action weights
            transformation_vector = np.vectorize(transform, otypes=[float])
            transformed_signal_action_weights = transformation_vector(
                self.signal_action_weights)
            # Each row represents a signal;
            # sum the transformed propensities to convert them into action proabilities given a signal
            row_sums = np.sum(transformed_signal_action_weights, axis=1)
            signal_action_probs = transformed_signal_action_weights.T / row_sums

            if world_state != -1:
                # we are peeking at the world state, and should consider additonal propensities

                # transform to workable weight scores
                transformed_state_action_weights = transformation_vector(
                    self.state_action_weights)
                # pick out given state
                given_state_action_weights = transformed_state_action_weights[world_state]
                # sum transformed propensities to convert to action probabilities
                state_action_sum = np.sum(given_state_action_weights)
                given_state_action_probs = given_state_action_weights / state_action_sum

                # linear combine signal-to-action probs for given signal with state-to-action probs
                signal_action_probs[:, signal] *= given_state_action_probs
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
        except ValueError:
            print("Passed a bad value to prob")
            pdb.set_trace()

        # record if needed
        if make_record:
            self.history.append(signal_action_probs.T)

        return action

    def update_action_weights(self, signal: int, action: int, reward: float, world_state: int) -> None:
        # Update the number of "balls in the urn" for a state, signal pair.
        # Generalizes reward to nearby states if stimulus_generalization == true during setup.

        # Give direct reward
        self.signal_action_weights[signal, action] += reward
        if world_state != -1:  # we peeked!
            self.state_action_weights[world_state, action] += reward

        # Stimulus generalization - often not used for receiver.
        if self.stimgen:
            # Two pointers ;)
            left_pointer = right_pointer = action
            for i in range(1, 4):  # arbitrary range. stimgen(0) = 1
                # calculate this step's reduced reward with the stimgen function defined in /util
                reduced_reward = stimgen(i) * reward

                right_pointer += 1
                if right_pointer < self.num_states:
                    # give reduced reward to ith right neighbor if in game bounds
                    self.signal_action_weights[signal,
                                               right_pointer] += reduced_reward
                    if world_state != -1:  # we peeked!
                        self.state_action_weights[world_state,
                                                  right_pointer] += reduced_reward

                left_pointer -= 1
                if left_pointer >= 0:
                    # give reduced reward to the ith left neighbor if in game bounds
                    self.signal_action_weights[signal,
                                               left_pointer] += reduced_reward
                    if world_state != -1:  # we peeked!
                        self.state_action_weights[world_state,
                                                  left_pointer] += reduced_reward

    def print_action_probs(self) -> None:
        # utility function for directly printing the probs from the most recent generate_action()

        # get probs from history
        probs = self.history[-1]

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
