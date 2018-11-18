#!/usr/bin/env python
"""Demonstration of the TD(0) reinforcement learning algorithm
described in Chapter 1 of the draft 2nd edition of Sutton and
Barto's book Reinforcement Learning: An Introduction.

Algorithm learns to play the Tic-Tac-Tie (noughts and crosses)
game. It can be trained against itself or against an expert
algorithm.
"""

# TODO:
# - for one-player games it doesn't need to use a game's
#    turn and player iterator attributes
# - how to update with multiple or continuous rewards?
# - no need to add initial values to value function
# - TD Learner does not need to memorize previous states
#   Can get them from the game
# - create a game.make_moves method
# - Are there proven ways to reduce learning_rate?
# - Allow player to be initialised from pickle file
# - Consider using property decorators
# - Can a neural network learn the value function?

import numpy as np
import itertools
from collections import deque
import random
import datetime
import pickle
from ast import literal_eval

__author__ = "Bill Tubbs"
__date__ = "October, 2018"
__version__ = "1.1"


class Player:
    """Base class for different types of game players.

    Attributes:
        Player.all_players (dict): Keeps a record of all player
                                   instances created.
    """

    all_players = {}

    def __init__(self, name):

        self.name = str(name)
        self.all_players[name] = self
        self.updates_on = True
        self.games_played = 0
        self.games_won = 0
        self.games_lost = 0

    def decide_next_move(self, game, role, show=False):
        """Override this method in your sub-class to return the
        player's next move in game assuming they are playing this
        role.

        Args:
            game (Game): Game being played.
            role (object): Role that the player is playing.
            show (bool): Print messages if True.

        Returns:
            move (tuple): Tuple containing (role, position).
        """

        raise NotImplementedError

        return move

    def make_move(self, game, role, show=False):
        """Ask player to make a move.  This method will call
        the player's decide_next_move method and will then
        execute the resulting move in the game.

        Args:
            game (Game): Game that player is playing.
            role (int): Role that the player is playing.
            show (bool): Print messages if True.
        """

        assert not game.game_over, "Can't make move. Game is over."

        # Player needs to trigger game start if method exists
        if len(game.moves) == 0:
            try:
                game.start()
            except AttributeError:
                pass

        move = self.decide_next_move(game, role, show)
        game.make_move(move)

    def update(self, game, reward, show=False):
        """Override this method if player needs rewards to learn
        how to play.

        Args:
            game (Game): Game that player is playing.
            reward (float): Reward value based on the last move
                made by player.
        """

        pass

    def update_terminal(self, game, reward, show=False):
        """Override this method if player needs rewards to learn
        how to play.

        Args:
            game (Game): Game that player is playing.
            reward (float): Terminal reward value based on the
                last move made by the winner.
        """

        pass

    def game_reset(self, game):
        """Override this method if player needs to do something
        when the game they are playing is reset.
        """

        pass

    def gameover(self, game, role, show=False):
        """Used to provide feedback to each player at the end of
        the game so they can learn from the results. If you over-
        ride this method in your sub-class, make sure to call
        super().gameover(game, role) so that the player keeps a
        record of games played, won, lost.

        Args:
            game (Game): Game being played.
            role (object): Role that the player is playing (could
                           be int or str depending on game).
            show (bool): Print a message (optional).
        """

        # Track number of games won and lost
        if game.game_over:
            if self.updates_on:
                self.games_played += 1
                if game.winner == role:
                    self.games_won += 1
                elif game.winner is not None:
                    self.games_lost += 1

    def save(self, filename=None):
        """
        Saves the player's current state as a pickle file. To reload
        a saved player object, use:

        >>> import pickle
        >>> my_player = pickle.load(open('Player 1.pkl', 'rb'))

        Args:
            filename (str): filename to use. If not provided, creates
                            filename from self.name. E.g. 'Player 1.pkl'
        """

        if filename is None:
            filename = self.name + '.pkl'

        pickle.dump(self, open(filename, 'wb'))

    def __repr__(self):

        return "Player(%s)" % self.name.__repr__()


class TicTacToeGame:
    """Simulates a game of tic tac toe (noughts and crosses).

    Class attributes:
        TicTacToeGame.name (str): The game's name ('Tic Tac Toe').
        TicTacToeGame.size (int): Width (and height) of board (3).
        roles [int, int]: The player roles ([1, 2]).
        TicTacToeGame.possible_n_players (list): List of allowed
            numbers of players ([2]).
        TicTacToeGame.marks (list): The characters used to represent
            each role's move on the board (['X', 'O']).
        TicTacToeGame.help_text (dict): Various messages (strings)
            to help user.
    """

    name = 'Tic Tac Toe'
    size = 3
    shape = (size, size)
    roles = [1, 2]
    possible_n_players = [2]
    marks = ['X', 'O']

    help_text = {
        'Move format': "row, col",
        'Move not available': "That position is not available.",
        'Number of players': "This game requires 2 players.",
        'Out of range': "Row and column must be in range 0 to %d." % (size - 1)
    }

    def __init__(self, moves=None):
        """Initialize a game.

        Args:
            moves (list): This is optional. Provide a list of completed
                moves. Each move should be a list or tuple of length 2
                where the first item is the player role and the second is
                the board position (row, col).
        """

        self.n_players = 2
        self.start_time = None
        self.end_time = None
        self.winner = None
        self.game_over = False
        self.player_iterator = itertools.cycle(self.roles)
        self.turn = next(self.player_iterator)
        self.state = np.zeros(self.shape, dtype='b')

        self.moves = []
        if moves is not None:
            for move in moves:
                self.make_move(move)
            self.start()

    def start(self):
        """Record start time (self.start_time)."""

        self.start_time = datetime.datetime.now()

    def stop(self):
        """Record end time (self.end_time)."""

        self.end_time = datetime.datetime.now()

    def reset(self):
        """Set the state of the game back to the beginning
        (no moves made).
        """

        self.moves = []
        self.player_iterator = itertools.cycle(self.roles)
        self.turn = next(self.player_iterator)
        self.state = np.zeros(self.shape, dtype='b')
        self.winner = None
        self.game_over = False
        self.start_time = None
        self.end_time = None

    def show_state(self):
        """Display the current state of the board."""

        chars = '_' + ''.join(self.marks)
        for row in self.state:
            print(" ".join(list(chars[i] for i in row)))

    def available_moves(self, state=None):
        """Returns list of available (empty) board positions (x, y).

        Args:
            state (np.ndarray): Array (size (3, 3)) of game state or if
                                not provided the current game state will
                                be used.
        """

        if state is None:
            state = self.state
        x, y = np.where(state == 0)

        return list(zip(x, y))

    def update_state(self, move, state=None):
        """Updates the game state with the move to be taken.

        Args:
            move (tuple): Tuple of length 2 containing the player role
                and the move (role, position). Position is also a tuple
                (row, col).
            state (np.ndarray): Array (size (3, 3)) of game state or if
                not provided the current game state will be used.

        Raises:
            AssertionError if the position is out of bounds or if
            there is already a move in that position.
        """

        role, position = move
        assert 0 <= position[0] < self.size, self.help_text['Out of range']
        assert 0 <= position[1] < self.size, self.help_text['Out of range']

        if state is None:
            state = self.state

        assert state[position] == 0, self.help_text['Move not available']
        state[position] = role

    def next_state(self, move, state=None):
        """Returns the next state of the game if move were to be
        taken from current game state or from state if provided.

        Args:
            move (tuple): Tuple of length 2 containing the player role
                and the move (role, position). Position is also a tuple
                (row, col).
            state (np.ndarray): Array (size (3, 3)) of game state or if
                not provided the current game state will be used.

        Returns:
            next_state (np.ndarray): copy of state after move made.
        """

        if state is None:
            state = self.state
        next_state = state.copy()
        self.update_state(move, state=next_state)

        return next_state

    def make_move(self, move, show=False):
        """Update the game state with a new move.

        Args:
            move (tuple): Tuple of length 2 containing the player role
                and the move (role, position). Position is also a tuple
                (row, col).
            show (bool): Print a message if True.
        """

        assert self.winner is None, "Player %s has already won" % \
                                    str(self.winner)

        role, position = move
        if role != self.turn:
            if role not in self.roles:
                raise ValueError("%d is not a valid player role." % role)
            else:
                raise ValueError("It is not player %d's turn." % role)

        self.update_state(move)
        self.moves.append(move)

        if show:
            print("Player %s made move %s" % (str(role), str(position)))

        self.check_if_game_over(role)
        if self.game_over:
            self.stop()
        self.turn = next(self.player_iterator)

    def reverse_move(self, show=False):
        """Reverse the last move made.

        Args:
            show (bool): Print a message if True.
        """

        last_move = self.moves.pop()
        self.state[last_move[1]] = 0
        if show:
            print("Last move reversed")
        self.check_if_game_over()
        self.turn = next(self.player_iterator)  # TODO: Only works for 2 player games!

    def get_rewards(self):
        """Returns any rewards at the current time step for players.
        In TicTacToe, there are no rewards until the end of the
        game so it sends a zero reward to each player after their
        opponent has made their move."""

        # TODO: Shouldn't really issue reward to 2nd player after first
        # move of game

        return {self.turn: 0.0}

    def get_terminal_rewards(self):
        """Returns the rewards at the end of the game for both players.
        In TicTacToe, there are no rewards until the end of the
        game but it sends a zero reward to each player after their
        opponent has made their move."""

        assert self.game_over

        if self.winner:

            # Winner's reward
            rewards = {self.winner: 1.0}

            # Loser's reward
            for role in [r for r in self.roles if r != self.winner]:
                rewards[role] = 0.0

        else:

            # Rewards for a draw
            rewards = {role: 0.5 for role in self.roles}

        return rewards

    def check_game_state(self, state=None, role=None):
        """Check the game state provided to see whether someone
        has won or if it is draw.

        Args:
            state (np.array): If not None, check if this game state
                array is a game-over state, otherwise check the
                actual game state (self.state).
            role (int): If specified, only check for a win by this
                game role.

        returns:
            game_over, winner (bool, bool): If there is a winner,
                winner will be the winning role. If the game is over,
                game_over will be True.
        """

        game_over, winner = False, None

        if state is None:
            state = self.state

        # If role specified, only check for a win by role
        if role:
            roles = [role]
        else:
            roles = self.roles

        # ~90% of execution time in this function
        # TODO: Ways to speed this up?  Call it less?
        # Idea: Could make a 3D array of win moves
        for role in roles:
            positions = (state == role)
            if any((
                    np.any(positions.sum(axis=0) == 3),
                    np.any(positions.sum(axis=1) == 3),
                    (np.diagonal(positions).sum() == 3),
                    (np.diagonal(np.fliplr(positions)).sum() == 3)
            )):
                game_over, winner = True, role
                break

        if np.sum(state == 0) == 0:
            game_over = True

        return game_over, winner

    def check_if_game_over(self, role=None):
        """Check to see whether someone has won or if it is draw.
        If the game is over, game_over will be set to True.
        If there is a winner, the attribute winner will be set
        to the winning role. This method is automatically called
        by make_move.

        Args:
            role (int): If specified, only check for a win by this
            game role.

        Returns:
            True if there is a winner else False.
        """

        self.game_over, self.winner = self.check_game_state(role=role)

        return self.game_over

    def generate_state_key(self, state, role):
        """Converts a game state (or afterstate) into a string of
        bytes containing characters that represent the following:
         '-': Empty board position
         'S': Position occupied by self
         'O': Position occupied by opponent

        This is used by TDLearner to create unique hashable keys
        for storing action-values in a dictionary.

        Example:
        > game.state
        array([[1, 0, 0],
               [2, 0, 0],
               [0, 0, 1]], dtype=int8)
        > game.generate_state_key(game.state, 1)
        b'S--O----S'

        Args:
            state (np.ndarray): Game state array.
            role (int): Player role.

        Returns:
            key (string): string of bytes representing game state.
        """

        if role == self.roles[0]:
            chars = ['-', 'S', 'O']
        elif role == self.roles[1]:
            chars = ['-', 'O', 'S']
        else:
            raise ValueError("Role does not exist in this game.")

        return np.array(chars, dtype='a')[state].tostring()

    def __repr__(self):

        params = []
        if self.moves:
            params.append("moves=%s" % self.moves.__repr__())

        return "TicTacToeGame(%s)" % ', '.join(params)


class HumanPlayer(Player):
    def __init__(self, name):
        """Player interface for human players.

        Args:
            name (str): Name to identify the player by.
        """
        super().__init__(name)

    def decide_next_move(self, game, role, show=True):
        """Determine next move in the game game by getting input
        from a human player.

        Args:
            game (Game): Game being played.
            role (object): Role that the player is playing (could
                           be int or str depending on game).
            show (bool): This has no effect. Messages are always
                         printed for human players.

        Returns:
            move (tuple): Tuple containing (role, position).
        """

        move_format = game.help_text['Move format']
        position = None
        while True:
            text = input("%s's turn (%s): " % (self.name, move_format))
            try:
                position = literal_eval(text)
            except (SyntaxError, ValueError):
                print("Move format is %s" % move_format)
                continue
            if not isinstance(position, tuple) or len(position) != 2:
                print("Move format is %s" % move_format)
                continue
            if position in game.available_moves():
                break
            print(game.help_text['Move not available'])
            print("Try again.")

        return role, position

    def gameover(self, game, role, show=True):

        super().gameover(game, role)

        if game.game_over:
            if show:
                if game.winner == role:
                    print("%s you won!" % self.name)
                elif game.winner is not None:
                    print("%s you lost!" % self.name)

    def __repr__(self):

        return "HumanPlayer(%s))" % self.name.__repr__()


class TDLearner(Player):
    def __init__(self, name="TD", learning_rate=0.25, gamma=1.0,
                 off_policy_rate=0.1, initial_value=0.5,
                 value_function=None, use_afterstates=True):
        """Tic-Tac-Toe game player that uses temporal difference (TD)
        learning algorithm.

        Args:
            name (str): Arbitrary name to identify the player
            learning_rate (float): Learning rate or step size (0-1).
            gamma (float): Discount rate (0-1).
            off_policy_rate (float): Frequency of off-policy actions
                (0-1).
            initial_value (float): Initial value to assign to new
                (unvisited) state.
            value_function (dict): Optionally provide a pre-trained
                value function.
        """

        super().__init__(name)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.off_policy_rate = off_policy_rate
        self.initial_value = initial_value
        if value_function is None:
            value_function = {}
        self.value_function = value_function
        self.saved_game_states = {}  # TODO: Should game save these?
        self.on_policy = None

    def get_value(self, state_key):
        """Returns a value from TDLearner's value_function for the
        game state represented by state_key. If there is no item for
        that move, returns the initial_value instead.
        """

        return self.value_function.get(state_key, self.initial_value)

    def save_state(self, game, state_key):
        """Adds action_key to a list of keys stored in dictionary
        self.saved_game_states for each game being played.
        """

        self.saved_game_states[game] = \
            self.saved_game_states.get(game, []) + [state_key]

    def decide_next_move(self, game, role, show=False):

        available_positions = game.available_moves()
        if len(available_positions) == 0:
            raise ValueError("There are no possible moves.")

        elif random.random() < self.off_policy_rate:
            # Random off-policy move
            self.on_policy = False
            position = random.choice(available_positions)
            next_state = game.next_state((role, position))
            next_state_key = game.generate_state_key(next_state, role)

        else:
            # On-policy move
            self.on_policy = True

            # Uses 'after-state' values
            options = []
            for position in available_positions:
                next_state = game.next_state((role, position))
                next_state_key = game.generate_state_key(next_state, role)
                action_value = self.get_value(next_state_key)
                options.append((action_value, position, next_state_key))

            max_value = max(options)[0]
            best_options = [m for m in options if m[0] == max_value]
            _, position, next_state_key = random.choice(best_options)

        # Save chosen state for learning updates later
        self.save_state(game, next_state_key)

        if show:
            move_format = game.help_text['Move format']
            print("%s's turn (%s): %s" % (self.name, move_format,
                                          str(position)))

        return role, position

    def get_saved_game_states(self, game):
        """Returns the dictionary self.saved_game_states.  If
        it doesn't exists, assigns an empty dictionary to it
        first.
        """

        states = self.saved_game_states.get(game, None)
        if states is None:
            states = []
            self.saved_game_states[game] = states

        return states

    def update(self, game, reward, show=False):
        """Update TDLearner's value function based on reward from
        game.

        Args:
            game (Game): Game that player is playing.
            reward (float): Reward value.
            show (bool): Print a message if True.
        """

        if self.updates_on and self.on_policy is True:

            # Retrieve previous actions in the game if
            # there were any
            states = self.get_saved_game_states(game)

            # Need at least 2 previous actions for a value update
            if len(states) > 1:

                # TD value function update
                self.value_function[states[-2]] = \
                    self.get_value(states[-2]) + self.learning_rate*(
                        reward + self.gamma*self.get_value(states[-1]) -
                        self.get_value(states[-2])
                    )

        if show:
            print("%s got %s reward." % (self.name, reward))

    def update_terminal(self, game, reward, show=False):
        """Update TDLearner's value function based on reward from
        game after terminal state reached.

        Args:
            game (Game): Game that player is playing.
            reward (float): Reward value.
            show (bool): Print a message if True.
        """

        if self.updates_on and self.on_policy is True:

            # Retrieve previous actions in the game if
            # there were any
            last_state = self.get_saved_game_states(game)[-1]

            # If game terminated then update last state value
            # as there are no future state-values
            self.value_function[last_state] = \
                self.get_value(last_state) + self.learning_rate*(
                    reward - self.get_value(last_state)
                )

        if show:
            print("%s got %s reward." % (self.name, reward))

    def game_reset(self, game):
        """Tells TD Learner that game has been reset.
        """

        # Delete stored list of previous game states
        del self.saved_game_states[game]

    def gameover(self, game, role, show=False):

        super().gameover(game, role)

        # Delete list of previous game states
        del self.saved_game_states[game]

    def copy(self, name):

        return TDLearner(name=name, learning_rate=self.learning_rate,
                         off_policy_rate=self.off_policy_rate,
                         value_function=self.value_function)

    def __repr__(self):

        return "TDLearner(%s)" % self.name.__repr__()


def winning_positions(game, role, available_positions=None, state=None):
    """Returns list of positions (row, col) that would result
    in player role winning if they took that position.

    Args:
        game (Game): Game that is being played.
        role (object): Role that the player is playing (could be
                       int or str depending on game).
        available_positions (list): List of positions to search (optional)
        state (np.ndarray): Game state array (shape may depend
                            on the game) of type int (optional).

    Returns:
        positions (list): List of winning positions
    """

    if available_positions is None:
        available_positions = game.available_moves(state=state)

    positions = []
    for position in available_positions:

        next_state = game.next_state((role, position), state=state)
        game_over, winner = game.check_game_state(next_state, role)
        if winner == role:
            positions.append(position)

    return positions


def fork_positions(game, role, available_positions, state=None):
    """Returns list of positions (row, col) where role has
    two opportunities to win (two non-blocked lines of 2) if
    they took that position.

    Args:
        game (Game): Game that is being played.
        role (object): Role that the player is playing (could be
                       int or str depending on game).
        available_positions (list): List of positions to search.
        state (np.ndarray): Game state array (shape may depend
                            on the game) of type int.

    Returns:
        positions (list): List of fork positions
    """

    positions = []
    for p1 in available_positions:
        next_state = game.next_state((role, p1), state=state)
        remaining_positions = game.available_moves(state=next_state)
        p2s = []
        for p2 in remaining_positions:
            state2 = game.next_state((role, p2), state=next_state)
            game_over, winner = game.check_game_state(state2, role)
            if winner == role:
                p2s.append(p2)
        if len(p2s) > 1:
            positions.append(p1)

    return positions


class ExpertPlayer(Player):
    """Optimal Tic-Tac-Toe game player that is unbeatable."""

    def __init__(self, name="EXPERT", seed=None):

        super().__init__(name)

        # Independent random number generator for sole use
        # by this instance
        self.rng = random.Random(seed)

    def decide_next_move(self, game, role, show=False):

        move_format = game.help_text['Move format']
        available_moves = game.available_moves()
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        center = (1, 1)
        opponent = game.roles[(game.roles.index(role) ^ 1)]

        move = None
        # 1. If first move of the game, play a corner or center
        if len(game.moves) == 0:
            move = (role, self.rng.choice(corners + [center]))

        if move is None:
            # 2. Check for winning moves
            positions = winning_positions(game, role,
                                          available_positions=available_moves)
            if positions:
                move = (role, self.rng.choice(positions))

        if move is None:
            # 3. Check for blocking moves
            positions = winning_positions(game, opponent,
                                          available_positions=available_moves)

            if positions:
                move = (role, self.rng.choice(positions))

        if move is None:
            # 4. Check for fork positions
            positions = fork_positions(game, role, available_moves)
            if positions:
                move = (role, self.rng.choice(positions))

        if move is None:
            # 5. Prevent opponent from using a fork position
            opponent_forks = fork_positions(game, opponent, available_moves)
            if opponent_forks:
                positions = []
                for p1 in available_moves:
                    next_state = game.next_state((role, p1))
                    p2s = winning_positions(game, role, state=next_state)
                    if p2s:
                        assert len(p2s) == 1, "Expert code needs debugging."
                        if p2s[0] not in opponent_forks:
                            positions.append(p1)
                if positions:
                    move = (role, self.rng.choice(positions))

        if move is None:
            # 6. Try to play center
            if center in available_moves:
                move = (role, center)

        if move is None:
            # 7. Try to play a corner opposite to opponent
            positions = []
            opposite_corners = {0: 3, 1: 2, 2: 1, 3: 0}
            for p1, p2 in opposite_corners.items():
                states = game.state[corners[p1]], game.state[corners[p2]]
                if states == (0, opponent):
                    positions.append(corners[p1])
            if positions:
                move = (role, self.rng.choice(positions))

        if move is None:
            # 8. Try to play any corner
            positions = [c for c in corners if game.state[c] == 0]
            if positions:
                move = (role, self.rng.choice(positions))

        if move is None:
            # 9. Play anywhere else - i.e. a middle position on a side
            move = (role, self.rng.choice(available_moves))

        if show:
            print("%s's turn (%s): %s" % (self.name, move_format, str(move)))

        return move

    def __repr__(self):

        return "ExpertPlayer(%s)" % self.name.__repr__()


class RandomPlayer(Player):
    def __init__(self, name="RANDOM", seed=None):
        """Tic-Tac-Toe game player that makes random moves.

        Args:
            name (str): Arbitrary name to identify the player
            seed (int): Random number generator seed.
        """

        super().__init__(name)

        # Independent random number generator for sole use
        # by this instance
        self.rng = random.Random(seed)

    def decide_next_move(self, game, role, show=False):
        available_moves = game.available_moves()
        move = (role, self.rng.choice(available_moves))

        if show:
            move_format = game.help_text['Move format']
            print("%s's turn (%s): %s" % (self.name, move_format, str(move)))

        return move

    def __repr__(self):
        return "RandomPlayer(%s)" % self.name.__repr__()


class GameController:
    """Manages one game instance with players."""

    def __init__(self, game, players, move_first=0, player_roles=None,
                 restart_mode='cycle'):
        """Setup a game.

        Args:
            game (Game): Game that is being played.
            players (list): List of player instances.
            move_first (int): Index of player to start first.
            player_roles (dict): Dictionary of players and their game
                roles.  If not provided, roles are assigned according
                to the order in the players list and the order of the
                game's roles list.
            restart_mode (string): Method to choose player roles when
                the game restarts. Options include 'cycle' which
                rotates player roles and 'random' to randomly assign
                roles to players.
        """

        self.game = game
        assert len(players) in game.possible_n_players, \
            game.help_text['Number of players']
        self.players = players
        if player_roles is None:
            player_roles = dict(zip(players, game.roles))
        self.player_roles = player_roles
        self.player_queue = deque(self.game.roles)
        self.restart_mode = restart_mode
        if restart_mode == 'random':
            random.shuffle(self.player_queue)
        elif restart_mode != 'cycle':
            raise ValueError("Game mode not valid.")
        self.player_queue.rotate(move_first)
        self.player_roles, self.players_by_role = self.player_role_dicts()

    def player_role_dicts(self):
        """Returns a tuple containing two dictionaries player_roles
        and players_by_role.  player_roles contains the role of
        each player in the game, and players_by_role contains
        the player in each game role.
        """

        return dict(zip(self.players, self.player_queue)), \
            dict(zip(self.player_queue, self.players))

    def announce_game(self):
        """Print a description of the game.

        Example:
        Game of Tic Tac Toe with 2 players ['Jack', 'Jill']
        """

        items = (
            self.game.name,
            len(self.players),
            str([p.name for p in self.players])
        )
        print("Game of %s with %d players %s" % items)

    def play(self, n_moves=None, show=True):
        """Play the game until game.game_over is True or after n
        moves if n > 0.

        Args:
            n_moves (int): Number of moves to play (optional).
            show (bool): Print messages if True.
        """

        assert self.game.game_over is not True, "Game is over. Use method " \
                                                "reset() to play again."

        if show:
            self.announce_game()

        # Main game loop
        while not self.game.game_over:

            if show:
                self.game.show_state()

            # Tell current player to make move
            player = self.players_by_role[self.game.turn]
            player.make_move(self.game, self.player_roles[player], show=show)

            # Get reward(s) from game
            rewards = self.game.get_rewards()

            # Send any rewards to players
            for role, reward in rewards.items():
                self.players_by_role[role].update(self.game, reward,
                                                  show=show)

            # Stop if limit on moves reached
            if n_moves is not None:
                n_moves -= 1
                if n_moves < 1:
                    break

        if self.game.game_over:
            # Get and send terminal rewards to players
            terminal_rewards = self.game.get_terminal_rewards()
            for role, reward in terminal_rewards.items():
                self.players_by_role[role].update_terminal(
                    self.game, reward, show=show)

            # Call gameover method of each player
            for player in self.players:
                player.gameover(self.game, self.player_roles[player],
                                show=show)
            if show:
                self.announce_result()

    def announce_result(self):
        """Print the game result and which player won."""

        self.game.show_state()
        if self.game.game_over:
            print("Game over!")
            if self.game.winner:
                print("%s won in %d moves" % (
                      self.players_by_role[self.game.winner].name,
                      len(self.game.moves)))
            else:
                print("Draw")

    def reset(self):

        if self.restart_mode == 'random':
            random.shuffle(self.player_queue)
        elif self.restart_mode == 'cycle':
            self.player_queue.rotate(1)
        self.player_roles, self.players_by_role = self.player_role_dicts()
        self.game.reset()

    def __repr__(self):

        return "GameController(%s, %s)" % (
            self.game.__repr__(),
            self.players.__repr__()
        )


def demo():
    """Simple demo of TicTacToeGame game dynamics.
    """

    game = TicTacToeGame()
    print("Game:", game)
    print("Marks:", game.marks)
    print("Roles:", game.roles)
    print("State:\n", game.state)

    print("Making some moves...")
    game.make_move((1, (0, 2)), show=True)
    game.make_move((2, (0, 1)), show=True)
    game.make_move((1, (1, 1)), show=True)
    game.make_move((2, (2, 2)), show=True)
    game.show_state()
    print("State:\n", game.state)

    print("Game over:", game.game_over)

    print("Moves so far:")
    print(game.moves)

    print("Turn:", game.turn)
    print("Available moves:", game.available_moves())

    game.make_move((1, (2, 0)), show=True)
    game.show_state()

    print("Game over:", game.game_over)
    print("Winner:", game.winner)

    game.reverse_move(show=True)
    game.show_state()

    print("Winner:", game.winner)

    print("Try player 2 move...")
    try:
        game.make_move((2, (1, 2)))
    except ValueError as err:
        print(err)

    print("Making some more moves...")
    game.make_move((1, (1, 2)), show=True)
    game.make_move((2, (2, 0)), show=True)
    game.make_move((1, (0, 0)), show=True)
    game.make_move((2, (1, 0)), show=True)
    game.show_state()
    game.make_move((1, (2, 1)), show=True)
    print("Game over:", game.game_over)
    print("Winner:", game.winner)


def tictactoe_game(players, move_first=0, show=True):
    """Demo of TicTacToeGame with two pre-defined players.

    Args:
        players (list): List of 2 Player instances.
        move_first (int): Specify which player should go first.
        show (bool): Print a message if True.
    """

    ctrl = GameController(TicTacToeGame(), players, move_first=move_first)
    ctrl.play(show=show)


def game_with_2_humans(names=("Player 1", "Player 2"), move_first=0,
                       n=1):
    """Demo of TicTacToeGame with two new human players.

    Args:
        names (list): A list containing two strings for names
            of the players (optional).
        move_first (int): Specify which player should go first.
        n (int or None): Number of games to play.  If n=None,
            it will loop indefinitely.
    """

    game = TicTacToeGame()
    players = [HumanPlayer(name) for name in names]
    play_looped_games(game, players, move_first=move_first, n=n)


def train_computer_players(players, iterations=1000, show=True):
    """Play repeated games with n computer players then play
    against one of them.

    Args:
        players (list): List of at least 2 Player instances.
        iterations (int): Number of iterations of training.
        show (bool): Print progress messages and results if True.
    """

    n_players = TicTacToeGame.possible_n_players[0]
    assert len(players) >= n_players, "Provide at least 2 players to train."

    stats = {p: {'won': 0, 'lost': 0, 'played': 0} for p in players}

    if show:
        print("\nTraining %d computer players..." % len(players))
    for i in range(iterations):
        game = TicTacToeGame()
        selected_players = random.sample(players, n_players)
        ctrl = GameController(game, selected_players)
        ctrl.play(show=False)
        for player in selected_players:
            stats[player]['played'] += 1
            if game.winner:
                if player == ctrl.players_by_role[game.winner]:
                    stats[player]['won'] += 1
                else:
                    stats[player]['lost'] += 1
        if show:
            if i % 100 == 0:
                print(i, "games completed")

    if show:
        print("\nResults:")
        for player in players:
            won, lost, played = (stats[player]['won'], stats[player]['lost'],
                                 stats[player]['played'])
            print("%s: won %d, lost %d, drew %d" % (player.name, won, lost,
                                                    played - won - lost))


def play_looped_games(game, players, move_first=0, n=None,
                      prompt=True, show=True):
    """Play repeated games between two players.  Displays a
    summary of results at the end.

    Args:
        game (Game): Game instance (for example, TicTacToeGame)
        players (list): List of 2 Player instances.
        move_first (int): Index of player to start first.
        n (int or None): Number of games to play.  If n=None,
            it will loop indefinitely.
        prompt (bool): If True, will prompt user each iteration
            with option to stop or play again.
        show (bool): Print messages if True.
    """

    ctrl = GameController(game, players, move_first=move_first)
    while True:
        print()
        ctrl.play(show=show)

        if n:
            n -= 1
            if n < 1:
                break

        if prompt:
            text = input("Press enter to play again or s to stop: ")
            if text.lower() == 's':
                break

        ctrl.reset()

    print("\nResults:")
    wins = 0
    for player in players:
        items = (player.name, player.games_won, player.games_played)
        print("Player %s won %d of %d games" % items)
        wins += player.games_won


def test_player(player, game=TicTacToeGame, seed=1):
    """
    Calculates a score based on the player's performance playing 100
    games of Tic Tac Toe, 50 against a random player and 50 against
    an expert. Score is calculated as follows:

    score = (1 - random_player.games_won/50)* \
            (random_player.games_lost/50)* \
            (1 - expert_player.games_won/50)

    An expert player should be able to get a score between 0.86 and
    1.0 (it's not possible to always win against a random player).

    Args:
        player (Player): Player instance.
        game (class): Class of game to use for the tests.
        seed (int): Random number generator seed. Changing this
            will change the test results slightly.

    Returns:
        score (float): Score between 0.0 and 1.0.
    """

    # Instantiate two computer opponents
    random_player = RandomPlayer(seed=seed)
    expert_player = ExpertPlayer(seed=seed)
    opponents = [random_player] * 50 + [expert_player] * 50

    # Shuffle with independent random number generator
    random.Random(seed).shuffle(opponents)

    game = game()
    player.updates_on, saved_mode = False, player.updates_on
    for i, opponent in enumerate(opponents):
        players = [player, opponent]
        ctrl = GameController(game, players, move_first=i % 2)
        ctrl.play(show=False)
        game.reset()
    player.updates_on = saved_mode

    score = (1 - random_player.games_won / 50) * \
            (random_player.games_lost / 50) * \
            (1 - expert_player.games_won / 50)

    return score


def main():
    """Code to demonstrate use of this module."""

    print("\nPlay Tic-Tac-Toe (Noughts and Crosses) against the"
          "computer algorithm.")
    computer_player = TDLearner("TD")
    name = input("Enter your name: ")
    human_player = HumanPlayer(name)
    n_iterations = 1000

    while True:

        # Train computer against itself
        # To do this you need to make a clone with the
        # same value function
        opponent = TDLearner("TD-clone")
        opponent.value_function = computer_player.value_function

        print("Computer is playing %d games against a clone of "
              "itself..." % n_iterations)
        train_computer_players([computer_player, opponent],
                               n_iterations)

        print("Now play against it.")
        game = TicTacToeGame()
        players = [human_player, computer_player]
        play_looped_games(game, players)

        # Slowly reduce the learning rate
        computer_player.learning_rate *= 0.9
        computer_player.off_policy_rate *= 0.9

        text = input("Press enter to do more training or q to quit: ")
        if text.lower() == 'q':
            break

if __name__ == "__main__":
    main()
