from collections.abc import Callable
import csv
import os
from .util import linear_reward_fn, normal_state_priors, display
from .agents import sender, receiver
import numpy as np


class SignalingGame:
    def __init__(self, n_states: int, n_signals: int, n_actions: int,
                 sn_stimgem: bool, rc_stimgen: bool, state_prior_dist: str,
                 observation_chance: float, reward_param: tuple[float, float],
                 reward_function: Callable) -> None:
        # Basic game constants
        self.num_states = n_states
        self.num_signals = n_signals
        self.num_actions = n_actions
        self.reward_parameter = reward_param

        self.current_state = 0
        self.current_signal = 0
        self.current_action = 0

        self.history = []

        # Variable/stochastic game constants
        self.null_signal = False
        self.rng = np.random.default_rng()
        self.observation_chance = np.array(
            [1-observation_chance, observation_chance])
        self.reward_function = reward_function(
            reward_param, self.null_signal)

        # Set prior distrobutions
        if state_prior_dist == "uniform":
            self.state_priors = np.full(
                self.num_states, 1 / self.num_states, dtype=np.float64)
        elif state_prior_dist == "normal":
            self.state_priors = normal_state_priors(self.num_states)
        else:
            raise RuntimeError("Only implemented ['normal', 'uniform']")

        # Initialize agents
        self.sender = sender.Sender(
            self.num_states, self.num_signals, self.rng, self.null_signal, sn_stimgem)

        self.receiver = receiver.Receiver(
            self.num_actions, self.num_signals, self.num_states, self.rng, rc_stimgen)

    def set_random_seed(self, seed: int):
        # for reproducability - not used during normal runs.
        self.rng = np.random.default_rng(seed)
        self.receiver.random = self.rng
        self.sender.random = self.rng

    def choose_state(self) -> int:
        # choose a random state, weighted by the priors.
        return self.rng.choice(self.num_states, p=self.state_priors)

    def roll_peek(self) -> bool:
        # "roll the dice" to choose if we can peek. 0 = no, 1 = yes.
        roll = self.rng.choice(2, p=self.observation_chance)
        return roll == 1

    def evaluate(self, state: int, action: int) -> float:
        return self.reward_function(state, action)

    def expected_payoff(self, step_signal_probs: np.ndarray, step_action_probs: np.ndarray) -> float:
        """Calculates the expected payoff given the probabilities of the Sender and the Receiver

        Args:
            signal_prob (np.ndarray): signal probabilities
            action_prob (np.ndarray): action probabilities

        Returns:
            float: the expected payoff
        """
        expected_payoff = 0
        for world_state in range(self.num_states):
            state_expected_payoff = 0
            n_signals_plus_null = self.num_signals + \
                (1 if self.null_signal else 0)
            for signal in range(n_signals_plus_null):
                state_signal_expected_payoff = 0

                # skip the null signal
                if self.null_signal and signal == self.num_signals:
                    continue

                for action in range(self.num_actions):
                    state_signal_expected_payoff += step_action_probs[signal, action] * self.evaluate(
                        world_state, action)
                state_expected_payoff += step_signal_probs[signal,
                                                           world_state] * state_signal_expected_payoff
            expected_payoff += state_expected_payoff
        return expected_payoff / self.num_states

    def info_measure_best_sig(self, signal_prob) -> tuple[list[float], list[int]]:
        # accomodate varied priors
        prob = np.zeros_like(signal_prob)
        for signal in range(self.num_signals):
            for state in range(self.num_states):
                prob[signal, state] = signal_prob[signal,
                                                  state] * self.state_priors[state]

        prob = (prob.T / np.sum(prob, axis=1)).T

        aggregate_info_measure = []
        aggregate_best_signals = []
        for state in range(self.num_states):
            state_info_measure = []
            for signal in range(self.num_signals):
                if self.null_signal and signal == self.num_signals:
                    continue
                # Note: given uniform state prior, prob[signal, state] * self.num_states = prob[signal, state] / P(state)
                signal_info_measure = prob[signal, state] * \
                    np.log(prob[signal, state] * self.num_states)
                state_info_measure.append(signal_info_measure)
            best_signal = np.argmax(state_info_measure)
            aggregate_info_measure.append(state_info_measure[best_signal])
            aggregate_best_signals.append(best_signal)
        return aggregate_info_measure, aggregate_best_signals

    def optimal_payoff(self) -> float:
        # I don't get it. I wish I did. mbarlow - 06/2025
        optimal_bucket_size = 2 * \
            (self.reward_parameter[0] // self.reward_parameter[1]) + 1
        expected_bucket_size = self.num_states // self.num_signals
        c, d = self.reward_parameter

        if self.null_signal and optimal_bucket_size < expected_bucket_size:
            return ((c * optimal_bucket_size)  # added parenthesis for redundancy.
                    - (d * (((optimal_bucket_size ** 2) - 1) / 4))) \
                * self.num_signals / self.num_states
        else:
            n_remainder_states = self.num_states % self.num_signals
            if expected_bucket_size % 2 == 0:
                return c - d * expected_bucket_size * \
                    (self.num_states + n_remainder_states) / (4 * self.num_states)
            else:
                return c - d * (expected_bucket_size + 1) * \
                    (self.num_states + n_remainder_states -
                     self.num_signals) / (4 * self.num_states)

    def optimal_info_measure(self) -> float:
        # TODO: Review this with someone who has a solid mathematical understanding of our setup
        optimal_bucket_size = 2 * \
            (self.reward_parameter[0] // self.reward_parameter[1]) + 1
        expected_n_null_used = self.num_states - \
            (self.num_signals * optimal_bucket_size)
        expected_bucket_size = self.num_states // self.num_signals
        n_remainder_states = self.num_states % self.num_signals

        if self.null_signal and expected_n_null_used > 0:
            return (optimal_bucket_size / self.num_states) * self.num_signals * np.log(self.num_states / optimal_bucket_size)
        else:
            return np.log(self.num_states) - \
                (n_remainder_states / self.num_signals) * np.log(expected_bucket_size + 1) - \
                (1 - n_remainder_states / self.num_signals) * \
                np.log(expected_bucket_size)

    def __call__(self, num_iters: int, record_interval: int, repeat_num: int, image_option: str) -> None:
        # Run the simulation

        # Main loop
        for step in range(num_iters):
            step_state = self.choose_state()
            self.current_state = step_state

            # determine if this step should be recorded. never record if interval <= -1
            should_record_step = (record_interval > 0 and (
                (step+1) % record_interval == 0))

            # Sender observes the world and chooses a signal to send
            step_signal = self.sender.generate_signal(
                step_state, should_record_step)
            self.current_signal = step_signal

            # Receiver observe signal and chooses and action.
            peek_state = -1
            if self.roll_peek():
                # if we can peek, send the world state. else, send -1
                peek_state = step_state
            step_action = self.receiver.generate_action(
                step_signal, peek_state, should_record_step)
            self.current_action = step_action

            # evaluate reward
            reward = self.evaluate(step_state, step_action)

            # send reward to sender and receiver
            self.sender.update_signal_weights(step_state, step_signal, reward)
            self.receiver.update_action_weights(
                step_signal, step_action, reward, peek_state)

            # save game state to history
            self.history.append({
                "state": step_state,
                "signal": step_signal,
                "step_action": step_action,
                "reward": reward
            })

        # record run info

        # strucutres to get+store data
        final_signal_probs = self.sender.history[-1]
        final_signal_action_probs = self.receiver.signal_action_history[-1]
        final_payoff = self.expected_payoff(
            final_signal_probs, final_signal_action_probs)

        null_usage_by_state = np.zeros(self.num_states, dtype=int)
        state_counts = np.zeros(self.num_states, dtype=int)

        # count null states if any
        if self.null_signal:
            for h in self.history:
                s = h["state"]
                sig = h["signal"]
                state_counts[s] += 1
                if sig == -1:
                    null_usage_by_state[s] += 1

        # generate visuals from game history
        output_dir = f"./simulations/{self.num_states}_{self.num_signals}_{self.num_actions}/"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.reward_parameter}{'_null' if self.null_signal else ''}_{num_iters}"
        if repeat_num is not None:
            filename += f"_{repeat_num}"
        if record_interval != -1 and image_option == "gif":
            display.gen_gif(
                self.sender.history,
                self.receiver.signal_action_history,
                self.receiver.state_action_history,
                self.expected_payoff,
                self.optimal_payoff(),
                self.info_measure_best_sig,
                self.optimal_info_measure(),
                num_iters,
                record_interval,
                duration=100,
                output_file=output_dir + filename + ".gif"
            )
        elif record_interval != -1 and image_option == "image":
            display.gen_single_heatmap(
                self.sender.history,
                self.receiver.signal_action_history,
                self.receiver.state_action_history,
                self.expected_payoff,
                self.optimal_payoff(),
                self.info_measure_best_sig,
                self.optimal_info_measure(),
                num_iters,
                record_interval,
                duration=100,
                output_file=output_dir + filename + ".jpg"
            )

        # print to csv
        csv_filename = "results_" + filename

        # assemble header list
        header_list = ["state", "null_count", "total_count", "null_fraction"]
        for i in range(self.num_signals-1):
            header_list.append(f"sn_sg_{i}")
        header_list.append("sn_sg_null")
        for i in range(self.num_actions):
            header_list.append(f"rc_ac_{i}")
        header_list.append("payoff")

        with open(output_dir + csv_filename + ".csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header_list)
            for s in range(self.num_states):
                if state_counts[s] > 0:
                    frac = null_usage_by_state[s] / state_counts[s]
                else:
                    frac = 0
                sg_prb = final_signal_probs[:, s]
                ac_prb = final_signal_action_probs[:, s]

                # assemble row
                row = [s, null_usage_by_state[s], state_counts[s], frac]
                for i in range(self.num_signals):
                    row.append(sg_prb[i])
                for i in range(self.num_actions):
                    row.append(ac_prb[i])
                row.append(final_payoff)

                writer.writerow(row)
