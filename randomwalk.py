#!/usr/bin/env python
"""Demonstration of TD reinforcement learning algorithms
described in Chapters 6 and 7 of the 2nd edition of Sutton
and Barto's book Reinforcement Learning: An Introduction.

Random Walk environment is used to test n-step TD and
TD-Lambda algorithms.
"""

import string
import numpy as np
from gamelearner import Environment


def calculate_true_values(game):
    """Returns a list of the true values of states in a
    RandomWalk game.  Note: these true values only apply
    when the agent acts randomly and when the discount
    factor is 1.0.
    """

    xp = [0, game.size + 1]
    fp = [-1.0, 1.0]

    true_values = np.interp(np.arange(game.size + 2), xp, fp)[1:-1]

    return true_values


def rms_error(values, true_values):
    """Root-mean-squared error of values compared to true values.
    """

    return np.sqrt(np.sum((np.array(values) -
                           np.array(true_values))**2)/len(values))


class RandomWalkGame(Environment):

    name = 'Random Walk'
    roles = [1]
    possible_n_players = [1]
    terminal_states = ['T1', 'T2']
    default_terminal_rewards = {'T1': 0.0, 'T2': 1.0}
    input_example = 'r'

    help_text = {
        'Move format': "'l' or 'r'",
        'Move not available': "That action is not available.",
        'Number of players': "This game is for 1 player."
    }

    def __init__(self, moves=None, size=5,
                 terminal_rewards=None):

        super().__init__(moves)

        self.size = size
        assert 1 < size <= 26  # States are labelled A-Z

        if terminal_rewards:
            self.terminal_rewards = terminal_rewards
        else:
            self.terminal_rewards = self.default_terminal_rewards

        # Create states
        self.states = [self.terminal_states[0]] \
            + list(string.ascii_uppercase[:size]) \
            + [self.terminal_states[1]]

        assert all([s in self.states for s in self.terminal_states])
        assert all([s in self.terminal_rewards for s in self.terminal_states])

        # Start in middle position
        self.start_state = self.states[self.size // 2 + 1]
        self.state = self.start_state

        # Define environment dynamics
        self.dynamics = {}
        for i in range(1, self.size + 1):
            s_left = self.states[i - 1]
            s_right = self.states[i + 1]
            self.dynamics[self.states[i]] = {'l': s_left, 'r': s_right}

        self.n_players = 1
        self.turn = 1
        self.winner = None

    def reset(self):

        super().reset()
        self.state = self.start_state
        self.winner = None

    def show_state(self, simple=False):

        if simple:
            print(self.state)
        else:
            # Displays the full random walk
            states_to_show = [self.state] + self.terminal_states
            print(' '.join((s if s in states_to_show else '_')
                           for s in self.states))

    def available_moves(self, state=None):

        if state is None:
            state = self.state

        return list(self.dynamics[state].keys())

    def next_state(self, state, move):

        role, action = move

        return self.dynamics[state][action]

    def update_state(self, move):

        self.state = self.next_state(self.state, move)

    def reverse_move(self, show=False):
        """Reverse the last move made.

        Args:
            show (bool): Print a message if True.
        TODO: Why do we need it?
        """

        raise NotImplementedError

    def get_rewards(self):
        """Returns any rewards at the current time step.  In
        RandomWalk, there are no rewards until the end of the
        game so send a zero reward."""

        return {1: 0.0}

    def get_terminal_rewards(self):
        """Returns the reward after the terminal state was
        reached."""

        assert self.game_over, "Game is not over"
        assert self.state in self.terminal_states

        return {1: self.terminal_rewards[self.state]}

    def check_game_state(self, state=None, role=None, calc=False):
        """Check the environment state to see if episode
        will terminate now.

        Args:
            state (np.array): If not None, check if this game state
                array is a game-over state, otherwise check the
                actual game state (self.state).
            role (int): If specified, only check for a win by this
                game role.
            calc (bool):

        returns:
            game_over, winner (bool, bool): If there is a winner,
                winner will be the winning role. If the game is over,
                game_over will be True.
        """

        if self.state in self.terminal_states:
            game_over, winner = True, 1
        else:
            game_over, winner = False, None

        return game_over, winner

    def generate_state_key(self, state, role):

        return state
