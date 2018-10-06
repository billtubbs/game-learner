#!/usr/bin/env python
"""Demonstration of the TD(0) reinforcement learning algorithm
described in Chapter 1 of the draft 2nd edition of Sutton and
Barto's book Reinforcement Learning: An Introduction.

Algorithm learns to play the Tic-Tac-Tie (noughts and crosses)
game. It can be trained against itself or against an expert
algorithm.
"""

# TODO:
# - create a game.make_moves method
# - Are there proven ways to reduce learning_rate?
# - Allow player to be initialised from pickle file
# - Consider using property decorators
# - Can a neural network learn the value function?
# - Ways to speed up check_game_state?

import numpy as np
import itertools
from collections import deque
import random
import datetime
import pickle
from ast import literal_eval

__author__ = "Bill Tubbs"
__date__ = "July, 2018"
__version__ = "1.0"


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
        self.updates = True
        self.games_played = 0
        self.games_won = 0
        self.games_lost = 0

    def decide_next_move(self, game, role, show=False):
        """Override this method in your sub-class to return the
        player's next move in game assuming they are playing this
        role.

        Args:
            game (Game): Game which is being played.
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
            game (Game): Game which player is playing.
            role (int): Role that the player is playing.
            show (bool): Print messages if True.
        """

        assert not game.game_over, "Can't make move. Game is over."

        if len(game.moves) == 0:
            game.start()
        move = self.decide_next_move(game, role, show)
        game.make_move(move)

    def reward(self, game, role, reward):
        """Override this method if playing needs rewards to learn
        how to play.

        Args:
            game (Game): Game which player is playing.
            role (int): Role that the player is playing.
            reward (float): Reward value based on the last move
                made by player.
        """

        pass

    def gameover(self, game, role, show=False):
        """Used to provide feedback to each player at the end of
        the game so they can learn from the results. If you over-
        ride this method in your sub-class, make sure to call
        super().gameover(game, role) so that the player keeps a
        record of games played, won, lost.

        Args:
            game (Game): Game which is being played.
            role (object): Role that the player is playing (could
                           be int or str depending on game).
        """

        # Track number of games won and lost
        if game.game_over:
            if self.updates:
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
        'Out of range': "Row and column must be in range 0 to %d." % (size-1)
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
        self.reset()
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
        """Set the state of the game to the beginning (no moves).
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
                          and the move (role, position). Position is
                          also a tuple (row, col).
            state (np.ndarray): Array (size (3, 3)) of game state or if
                                not provided the current game state will
                                be used.

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
                          and the move (role, position). Position is
                          also a tuple (row, col).
            state (np.ndarray): Array (size (3, 3)) of game state or if
                                not provided the current game state will
                                be used.

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
                          and the move (role, position). Position is
                          also a tuple (row, col).
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

    def show_moves(self):
        """Show a list of the moves made.
        """

        for i, move in enumerate(self.moves, start=1):
            print(i, move)

    def get_rewards(self):
        """Returns the rewards at the current time step. For
        TicTacToe, there are no rewards until game is over."""

        rewards = []
        if self.game_over:
            if self.winner:

                # Winner's reward
                rewards.append((self.winner, 1.0))

                # Loser's reward
                for role in [r for r in self.roles if r != self.winner]:
                    rewards.append((role, 0.0))

            else:

                # Rewards for a draw
                for role in self.roles:
                    rewards.append((role, 0.5))

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
        # TODO: Ways to speed this up?
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
        """Converts a game state in the form of an array into a
        string of bytes containing characters that represent the
        following:
         '-': Empty board position
         'S': Position occupied by self
         'O': Position occupied by opponent

        This is used by TDLearner to create unique hashable keys
        for storing values in a dictionary.

        Example:
        > game.state
        array([[1, 0, 0],
               [2, 0, 0],
               [0, 0, 1]], dtype=int8)
        > game.generate_state_key(game.state, 1)
        b'S--O----S'

        Args:
            state (np.ndarray): Game state array (shape may depend
                                on the game) of type int.
            role (object): Role that the player is playing (could
                           be int or str depending on game).

        Returns:
            key (string): string of bytes representing game state.
        """

        # TODO: This only works for two-player games
        if role == self.roles[0]:
            chars = ['-', 'S', 'O']
        elif role == self.roles[1]:
            chars = ['-', 'O', 'S']
        else:
            raise NotImplementedError("Role not found for this game.")

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
            game (Game): Game which is being played.
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

    def __init__(self, name="TD", learning_rate=0.25,
                 off_policy_rate=0.1, default_value=0.5,
                 value_function=None):
        """Tic-Tac-Toe game player that uses temporal difference (TD)
        learning algorithm.

        Args:
            name (str): Arbitrary name to identify the player
            learning_rate (float): Learning rate or step size (0-1).
            off_policy_rate (float): Frequency of off-policy actions
                (0-1).
            default_value (float): Initial value to assign to new
                (unvisited) state.
            value_function (dict): Optionally provide a pre-trained
                value function.
        """

        super().__init__(name)

        self.default_value = default_value
        if value_function is None:
            value_function = {}
        self.value_function = value_function
        self.learning_rate = learning_rate
        self.off_policy_rate = off_policy_rate
        self.previous_states = dict()

    def decide_next_move(self, game, role, show=False):

        move_format = game.help_text['Move format']

        available_moves = game.available_moves()
        if len(available_moves) == 0:
            raise ValueError("There are no possible moves.")

        elif random.random() < self.off_policy_rate:
            # Random off-policy move
            position = random.choice(available_moves)
            next_state = game.next_state((role, position))
            key = game.generate_state_key(next_state, role)
            if key not in self.value_function:
                self.value_function[key] = self.default_value

        else:
            # On-policy learning
            move_values = []
            for position in available_moves:
                next_state = game.next_state((role, position))
                key = game.generate_state_key(next_state, role)
                value = self.value_function.get(key, None)
                if value is None:
                    value = self.default_value
                    self.value_function[key] = value
                move_values.append((value, position, key))

            max_value = max(move_values)[0]
            best_moves = [move for move in move_values if move[0] == max_value]
            value, position, key = random.choice(best_moves)

            # Update value function if on-policy and learning is True
            if self.updates:
                if game in self.previous_states:
                    previous_key = self.previous_states[game]
                    self.value_function[previous_key] = (
                        (1 - self.learning_rate) *
                        self.value_function[previous_key] +
                        self.learning_rate * value
                    )

        self.previous_states[game] = key

        if show:
            print("%s's turn (%s): %s" % (self.name, move_format,
                                          str(position)))

        return role, position

    def reward(self, game, role, reward):
        """Send TDLearner a reward after each move.

        Args:
            game (Game): Game which player is playing.
            role (int): Role that the player is playing.
            reward (float): Reward value based on the last move
                made by player.
        """

        previous_key = self.previous_states[game]
        self.value_function[previous_key] = reward

    def gameover(self, game, role, show=False):

        super().gameover(game, role)

        # Delete record of previous state
        del self.previous_states[game]

    def copy(self, name):

        return TDLearner(name=name, learning_rate=self.learning_rate,
                         off_policy_rate=self.off_policy_rate,
                         value_function=self.value_function)

    def __repr__(self):

        return "TDLearner(%s)" % self.name.__repr__()


def winning_positions(game, role, available_moves, state=None):
    """Returns list of positions (row, col) that would result
    in player role winning if they took that position.

    Args:
        game (Game): Game that is being played.
        role (object): Role that the player is playing (could be
                       int or str depending on game).
        available_moves (list): List of positions to search
        state (np.ndarray): Game state array (shape may depend
                            on the game) of type int.

    Returns:
        positions (list): List of winning positions
    """

    positions = []
    for position in available_moves:
        next_state = game.next_state((role, position), state=state)
        game_over, winner = game.check_game_state(next_state, role)
        if winner == role:
            positions.append(position)

    return positions


def fork_positions(game, role, available_moves, state=None):
    """Returns list of positions (row, col) where role has
    two opportunities to win (two non-blocked lines of 2) if
    they took that position.

    Args:
        game (Game): Game that is being played.
        role (object): Role that the player is playing (could be
                       int or str depending on game).
        available_moves (list): List of positions to search
        state (np.ndarray): Game state array (shape may depend
                            on the game) of type int.

    Returns:
        positions (list): List of fork positions
    """

    positions = []
    for p1 in available_moves:
        next_state = game.next_state((role, p1), state=state)
        remaining_positions = game.available_moves(next_state)
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

    def __init__(self, name="EXPERT"):

        super().__init__(name)

    def decide_next_move(self, game, role, show=False):

        move_format = game.help_text['Move format']
        available_moves = game.available_moves()
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        center = (1, 1)
        opponent = game.roles[(game.roles.index(role) ^ 1)]

        move = None
        # 1. If first move of the game, play a corner or center
        if len(game.moves) == 0:
            move = (role, random.choice(corners + [center]))

        if move is None:
            # 2. Check for winning moves
            positions = winning_positions(game, role, available_moves)
            if positions:
                move = (role, random.choice(positions))

        if move is None:
            # 3. Check for blocking moves
            positions = winning_positions(game, opponent, available_moves)

            if positions:
                move = (role, random.choice(positions))

        if move is None:
            # 4. Check for fork positions
            positions = fork_positions(game, role, available_moves)
            if positions:
                move = (role, random.choice(positions))

        if move is None:
            # 5. Prevent opponent from using a fork position
            opponent_forks = fork_positions(game, opponent, available_moves)
            if opponent_forks:
                positions = []
                for p1 in available_moves:
                    next_state = game.next_state((role, p1))
                    p2s = winning_positions(game, role, available_moves,
                                            next_state)
                    if p2s:
                        assert len(p2s) == 1, "Expert code needs debugging."
                        if p2s[0] not in opponent_forks:
                            positions.append(p1)
                if positions:
                    move = (role, random.choice(positions))

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
                move = (role, random.choice(positions))

        if move is None:
            # 8. Try to play any corner
            positions = [c for c in corners if game.state[c] == 0]
            if positions:
                move = (role, random.choice(positions))

        if move is None:
            # 9. Play anywhere else - i.e. a middle position on a side
            move = (role, random.choice(available_moves))

        if show:
            print("%s's turn (%s): %s" % (self.name, move_format, str(move)))

        return move

    def __repr__(self):

        return "ExpertPlayer(%s)" % self.name.__repr__()


class RandomPlayer(Player):

    def __init__(self, name="RANDOM", seed=1):
        """Tic-Tac-Toe game player that makes random moves.

        Args:
            name (str): Arbitrary name to identify the player
            seed (int): Random number generator seed.
        """

        super().__init__(name)

        # Independent random number generator for sole use
        # by this instance
        self.rng = np.random.RandomState(seed)

    def decide_next_move(self, game, role, show=False):

        move_format = game.help_text['Move format']
        available_moves = game.available_moves()
        idx = self.rng.choice(len(available_moves))
        move = (role, available_moves[idx])

        if show:
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
        self.set_player_roles()

    def set_player_roles(self):

        self.player_roles = dict(zip(self.players, self.player_queue))
        self.players_by_role = dict(zip(self.player_queue, self.players))

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
        """Play the game until game.game_over is True or after
        n moves if n > 0.

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

            player = self.players_by_role[self.game.turn]
            player.make_move(self.game, self.player_roles[player], show=show)

            rewards = self.game.get_rewards()

            for role, reward in rewards:
                self.players_by_role[role].reward(self.game, role, reward)

            # Stop if limit on moves reached
            if n_moves is not None:
                n_moves -= 1
                if n_moves < 1:
                    break

        if self.game.game_over:
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
                print("%s won" % self.players_by_role[self.game.winner].name)
            else:
                print("Draw")

    def reset(self):

        if self.restart_mode == 'random':
            random.shuffle(self.player_queue)
        elif self.restart_mode == 'cycle':
            self.player_queue.rotate(1)
        self.set_player_roles()
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
    game.show_moves()

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
                       repeat=True):
    """Demo of TicTacToeGame with two new human players.

    Args:
        names (list): A list containing two strings for names
                      of the players (optional)
        move_first (int): Specify which player should go first.
    """

    game = TicTacToeGame()
    players = [HumanPlayer(name) for name in names]
    if repeat:
        n = None
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
        n (int or None): Number of games to play.  If n=None,
            it will loop indefinitely.
        prompt (bool): If True, will prompt user each iteration
            with option to stop or play again.
    """

    ctrl = GameController(game, players, move_first=move_first)
    while True:
        print()
        ctrl.play(show=show)

        if n:
            n = n - 1
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
    random_player = RandomPlayer(seed)
    expert_player = ExpertPlayer()
    opponents = [random_player]*50 + [expert_player]*50

    # Shuffle with independent random number generator
    np.random.RandomState(seed).shuffle(opponents)

    game = game()
    player.updates, saved_mode = False, player.updates
    for i, opponent in enumerate(opponents):
        players = [player, opponent]
        ctrl = GameController(game, players, move_first=i % 2)
        ctrl.play(show=False)
        game.reset()
    player.updates = saved_mode

    score = (1 - random_player.games_won/50)* \
            (random_player.games_lost/50)* \
            (1 - expert_player.games_won/50)

    return score


def main():
    """Code to demonstrate use of this module."""

    n = 5
    print("\nPlay Tic-Tac-Toe (Noughts and Crosses) against %d\n"
          "trained computer algorithms." % n)
    computer_players = [TDLearner("TD%02d" % i) for i in range(n)]

    name = input("Enter your name: ")
    human_player = HumanPlayer(name)

    while True:
        train_computer_players(computer_players, 1000)

        best_wins = max([p.games_won for p in computer_players])
        best_players = [p for p in computer_players if
                        p.games_won == best_wins]
        if len(best_players) > 1:
            best_losses = min([p.games_lost for p in best_players])
            best_players = [p for p in best_players if
                            p.games_lost == best_losses]
            if len(best_players) > 1:
                most_played = max([p.games_played for p in best_players])
                best_players = [p for p in best_players if
                                p.games_played == most_played]
        best_player = random.choice(best_players)
        print("Best player so far:", best_player)

        game = TicTacToeGame()
        players = [human_player, best_player]
        play_looped_games(game, players)
        for p in computer_players:
            # Slowly reduce the learning rate
            p.learning_rate *= 0.9
            p.off_policy_rate *= 0.9
        text = input("Press enter to do more training or q to quit: ")
        if text.lower() == 'q':
            break

        computer_players = [best_player.copy("TD%02d" % i) for i in range(n)]
        print("%d clones of %s made" % (n-1, str(best_player)))


if __name__ == "__main__":
    main()
