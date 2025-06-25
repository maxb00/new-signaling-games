import numpy as np
import util
from agents.sender import Sender
from agents.receiver import Receiver


class SignalingGame:
    def __init__(self, n_states: int, n_signals: int, n_actions: int, sn_stimgem: bool, rc_stimgen: bool, state_prior_dist: str, peek_chance: float) -> None:
        # Basic game constants
        self.num_states = n_states
        self.num_signals = n_signals
        self.num_actions = n_actions

        self.current_state = 0
        self.current_signal = 0
        self.current_action = 0

        self.peek_chance = np.array([1-peek_chance, peek_chance])

        self.history = []

        # Variable/stochastic game constants
        self.null_signal = False
        self.rng = np.random.default_rng()

        # Set prior distrobutions
        if state_prior_dist == "uniform":
            self.state_priors = np.full(
                self.num_states, 1 / self.num_states, dtype=np.float64)
        elif state_prior_dist == "normal":
            self.state_priors = util.normal_state_priors(self.num_states)
        else:
            raise RuntimeError("Only implemented ['normal', 'uniform']")

        # Initialize agents
        self.sender = Sender(
            self.num_states, self.num_signals, self.rng, self.null_signal, sn_stimgem)

        self.receiver = Receiver(
            self.num_actions, self.num_signals, self.num_states, self.rng, rc_stimgen)

    def set_random_seed(self, seed: int):
        # for reproducability - not used during normal runs.
        self.random = np.random.default_rng(seed)
        self.receiver.random = self.random
        self.sender.random = self.random

    def choose_state(self) -> int:
        # choose a random state, weighted by the priors.
        return self.random.choice(self.num_states, p=self.state_priors)

    def roll_peek(self) -> bool:
        # choose if we can peek. 0 = no, 1 = yes.
        roll = self.random.choice(2, p=self.peek_chance)
        return roll == 1

    def __call__(self, num_iters: int, record_interval: int, repeat_num: int) -> None:
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
