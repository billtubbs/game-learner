#!/usr/bin/env python
"""Implementation of the game Connect 4 (by Hasbro) for testing
reinforcement learning algorithms.
"""

import random
import numpy as np
import itertools
from gamelearner import Environment, Player


class Connect4Game(Environment):
    """Simulates a game of Connect 4.

    Class attributes:
        Connect4Game.name (str): The game's name ('Connect 4').
        Connect4Game.shape (int): Width and height of board (6, 7).
        roles [int, int]: The player roles ([1, 2]).
        Connect4Game.possible_n_players (list): List of allowed
            numbers of players ([2]).
        Connect4Game.marks (list): The characters used to represent
            each role's move on the board (['S', 'O']).
        Connect4Game.connect (int): Number of discs in a row to win (4).
        Connect4Game.help_text (dict): Various messages (strings)
            to help user.
    """

    name = 'Connect 4'
    shape = (6, 7)
    roles = [1, 2]
    possible_n_players = [2]
    marks = ['X', 'O']
    connect = 4
    terminal_rewards = {'win': 1.0, 'lose': 0.0, 'draw': 0.5}
    input_example = 0

    help_text = {
        'Move format': "column from left",
        'Move not available': "That position is not available.",
        'Number of players': "This game requires 2 players.",
        'Out of range': "slot must be in range 0 to %d." % (shape[1] - 1)
    }

    # Data objects for analyzing board
    _steps = {
        'u': (1, 0),
        'd': (-1, 0),
        'r': (0, 1),
        'l': (0, -1),
        'ur': (1, 1),
        'dr': (-1, 1),
        'ul': (1, -1),
        'dl': (-1, -1)
    }

    # Function used by _check_positions method
    _fcum = lambda x1, x2: (x1 + x2)*x2

    def __init__(self, moves=None):
        """Initialize a game.
        Args:
            moves (list): This is optional. Provide a list of completed
                moves. Each move should be a list or tuple of length 2
                where the first item is the player role and the second is
                the board position (col).
        """
        self.n_players = 2
        self._board_full, self._state = self._empty_board_state()
        self._fill_levels = np.zeros(self.shape[1], dtype='int8')
        self._pos_last = None
        self.winner = None
        self.player_iterator = itertools.cycle(self.roles)
        self.turn = next(self.player_iterator)
        super().__init__(start_state=self._state, moves=moves)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state[:] = state

    def _empty_board_state(self):
        """Initialize board_full and state."""
        # board_full has a border set to -1
        board_full = -np.ones(np.array(self.shape) + (2, 2), dtype='int8')
        state = self.state_from_board_full(board_full)
        state[:] = 0
        return board_full, state

    def state_from_board_full(self, board_full):
        return board_full[1:1+self.shape[0], 1:1+self.shape[1]]

    def reset(self):
        """Set the state of the game back to the beginning
        (no moves made).
        """
        super().reset()
        self._board_full, self._state = self._empty_board_state()
        self._fill_levels = np.zeros(self.shape[1], dtype='int8')
        self._pos_last = None
        self.player_iterator = itertools.cycle(self.roles)
        self.turn = next(self.player_iterator)
        self.winner = None

    def show_state(self):
        """Display the current state of the board."""

        chars = '_' + ''.join(self.marks)
        for row in reversed(self.state):
            print(" ".join(list(chars[i] for i in row)))

    @staticmethod
    def _get_fill_levels(state):
        # Note: This assumes proper filling!
        return (state > 0).sum(axis=0)

    def available_moves(self, state=None):
        """Returns list of available (empty) moves (slots).
        Args:
            state (np.ndarray): Array (shape (6, 7)) of game state or if
                                not provided the current game state will
                                be used.
        """

        if state is None:
            state = self.state
            spaces_left = self._fill_levels < self.shape[0]
        else:
            spaces_left = self._get_fill_levels(state) < self.shape[0]

        return np.nonzero(spaces_left)[0]

    def _get_neighbours(self, pos, board_full=None):
        #TODO: Is this method actually used?
        if board_full is None:
            board_full = self._board_full
        neighbours = {d: board_full[(step[0]+pos[0], step[1]+pos[1])]
                      for d, step in self._steps.items()}
        return neighbours

    def _chain_in_direction(self, position, direction, role, board_full=None):
        """Finds number of matching discs in one direction."""
        if board_full is None:
            board_full = self._board_full
        step = self._steps[direction]
        for i in range(self.connect):
            position = (step[0]+position[0], step[1]+position[1])
            x = board_full[position]
            if x != role:
                break
        return i

    def _check_game_state_from_position(self, position, role, board_full=None):
        results = {}
        for direction, step in self._steps.items():
            n = self._chain_in_direction(position, direction, role, board_full)
            if n == self.connect - 1:
                return True
            results[direction] = n
        for d1, d2 in [('u', 'd'), ('l', 'r'), ('ul', 'dr'), ('dl', 'ur')]:
            if results[d1] + results[d2] >= self.connect-1:
                return True
        return False

    def _check_game_state_after_move(self, move, board_full=None):
        if board_full is None:
            board_full = self._board_full
            fill_levels = self._fill_levels
        else:
            state = self.state_from_board_full(board_full)
            fill_levels = self._get_fill_levels(state)
        role, column = move
        level = fill_levels[column]
        assert level < self.shape[0]
        position = (level+1, column+1)
        assert board_full[position] == 0
        return self._check_game_state_from_position(position, role,
                                                    board_full=board_full)

    @staticmethod
    def _next_available_position(state, col):
        # Note: This assumes proper filling!
        return (state[:, col] > 0).sum()

    def _check_positions(self, positions, connect=None):
        """Check bool array positions for a connect x."""
        if connect is None:
            connect = self.connect
        _fcum = lambda x1, x2: (x1 + x2)*x2
        positions = positions.astype('int8')
        temp = np.empty_like(positions)
        temp[:] = list(itertools.accumulate(positions, _fcum))
        max_vert = temp.max()
        temp.T[:] = list(itertools.accumulate(positions.T, _fcum))
        max_horiz = temp.max()
        diagonals = [np.diagonal(positions, offset=k) for k in range(-2, 4)]
        max_diag = max(max(itertools.accumulate(x, _fcum)) for x in diagonals)
        return max(max_horiz, max_vert, max_diag) >= connect

    def _check_game_state_for_winner(self, state, role=None):

        # If role specified, only check for a win by role
        roles = self.roles if role is None else [role]

        # Check whole state
        winner = None
        for role in roles:
            positions = (state == role)
            if self._check_positions(positions):
                winner = role
                break

        return winner

    def check_game_state(self, state=None, role=None):
        """Check the game state to see if it will terminate now.

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
            if self.moves:
                # Check state from position of last move only
                # No need to check rest of board as long as it
                # was checked after every previous move.
                role = self.moves[-1][0]
                position = (self._pos_last[0]+1, self._pos_last[1]+1)
                game_over = self._check_game_state_from_position(position, role)
                if game_over:
                    winner = self.moves[-1][0]
                else:
                    winner = None
                    # Check for a draw
                    # This is equivalent to
                    # if np.all(self.state > 0):
                    if len(self.moves) == self.shape[0]*self.shape[1]:
                        game_over = True
            else:
                game_over, winner = False, None
            return game_over, winner

        # If state was provided as an argument then need to
        # check whole state for winner
        winner = self._check_game_state_for_winner(state, role=role)
        if winner is not None:
            game_over = True
        else:
            # Check for a draw
            if np.all(state > 0):
                game_over = True

        return game_over, winner

    def next_state(self, state, move, role_check=True):
        """Returns the next state of the game when move is
        taken from current game state or from state if
        provided.

        Args:
            state (np.ndarray): Array (shape (6, 7)) of board state
                or if not provided the current game state will be
                used.
            move (tuple): Tuple of length 2 containing the player
                role and the move (role, position). Position is also
                a tuple (row, col).
            role_check (bool): If True, checks to make sure it is role's
                turn.
        Returns:
            next_state (np.ndarray): copy of state after move made.
        Raises:
            ValueError if it is not role's turn.
            AssertionError if the position is out of bounds or if
            there is already a move in that position.
        """

        role, position = move
        if role_check:
            if role != self.turn:
                raise ValueError(f"It is not player {role}'s turn.")

        assert 0 <= position < self.shape[1], self.help_text['Out of range']
        fill_level = self._fill_levels[position]
        assert fill_level < self.shape[0], self.help_text['Move not available']

        next_state = state.copy()
        next_state[fill_level, position] = role

        return next_state

    def make_move(self, move, show=False):
        """Update the game state with a new move.
        Args:
            move (tuple): Tuple of length 2 containing a
                player role and action (role, action).
            show (bool): Print a message if True.
        """
        position = move[1]
        fill_level = self._fill_levels[position]
        self._pos_before_last = self._pos_last
        self._pos_last = (fill_level, position)
        try:
            super().make_move(move, show)
        except ValueError as err:
            # If error raised (e.g. not players turn) need
            # to reverse assignment to self._pos_last
            self._pos_last = self._pos_before_last
            raise err
        self._fill_levels[position] += 1
        self.turn = next(self.player_iterator)

    def reverse_move(self, show=False):
        """Reverse the last move made.

        Args:
            show (bool): Print a message if True.
        """
        self.moves.pop()
        self.state[self._pos_last] = 0  # Removes last disc from board
        self._fill_levels[self._pos_last[1]] -= 1
        if len(self.moves) > 0:
            last_position = self.moves[-1][1]
            self._pos_last = (self._fill_levels[last_position] - 1,
                              last_position)
        else:
            self._pos_last = None
        # TODO: This only works for 2 player games:
        self.turn = next(self.player_iterator)
        self.check_if_game_over()

    def get_rewards(self):
        """Returns any rewards at the current time step for
        players.  In Connect 4, there are no rewards until the
        end of the game so it sends a zero reward to each
        player after the opponent has made their move.
        """

        # TODO: Shouldn't really issue reward to 2nd player after first
        # move of game

        return {self.turn: 0.0}

    def get_terminal_rewards(self):
        """Returns the rewards at the end of the game for both
        players.
        """

        assert self.game_over

        if self.winner:

            # TODO: Last player to move should get reward from get_rewards().
            # Only the other player needs a special way to get their reward.

            # Winner's reward
            rewards = {self.winner: self.terminal_rewards['win']}

            # Loser's reward
            for role in [r for r in self.roles if r != self.winner]:
                rewards[role] = self.terminal_rewards['lose']

        else:

            # Rewards for a draw
            rewards = {role: self.terminal_rewards['draw'] for role in self.roles}

        return rewards

    def generate_state_key(self):
        raise NotImplementedError()


def wins_from_next_move(game, role, board_full=None, moves=None):
    if board_full is None:
        board_full = game._board_full
        state = game.state
    else:
        state = game.state_from_board_full(board_full)
    if moves is None:
        moves = game.available_moves(state)
    wins = {}
    fill_levels = game._get_fill_levels(state)
    for col in moves:
        pos_fb = (fill_levels[col]+1, col+1)
        win = game._check_game_state_from_position(pos_fb, role, board_full=board_full)
        wins[col] = win
    return wins


def check_for_obvious_move(game, role, board_full=None, state=None,
                           fill_levels=None,
                           terminal_values={'win': 1, 'loss': -1, 'draw':0},
                           depth=1):
    """Analyses the current board state (or board_full if
    provided) from the perspective of the player role.

    Returns
        value, positions (float, list): value of the current
            state if it is a terminal state (from terminal_values)
            else None, and a list of best positions (columns) to
            play on next move.
    """
    if board_full is None:
        board_full = game._board_full
        state = game.state
        fill_levels = game._fill_levels
    else:
        state = game.state_from_board_full(board_full)
    if fill_levels is None:
        fill_levels = game._get_fill_levels(state)

    # Check if board already full (draw)
    # (This function should not be called in this case)
    if fill_levels.sum() == game.shape[0]*game.shape[1]:
        raise ValueError("No available moves")

    opponent = role ^ 3
    win_value, loss_value = (terminal_values['win'],
                                terminal_values['loss'])

    # TODO: This should not be in this func
    # 0. Check if early move of game
    #if n_moves == 0:
    #    return None, [3]
    #elif (n_moves == 1) and (state[0,3] == opponent):
    #    return None, [3]

    # 1. Check for a win by role on next move
    possible_moves = wins_from_next_move(game, role, board_full=board_full)
    n_wins = sum(possible_moves.values())
    if n_wins > 0:
        winning_moves = [col for col, win in possible_moves.items() if win]
        return win_value, winning_moves

    if len(possible_moves) == 1:
        # 2. Check if draw (last move but no win)
        if fill_levels.sum() == game.shape[0]*game.shape[1] - 1:
            return terminal_values['draw'], list(possible_moves.keys())

    #TODO: Could continue deeper search if only one move possible
    if depth > 0:
        # 3. Check what opponent could do next for each possible move
        bf2 = board_full.copy()  # TODO: Could remove if sure it is restored
        state = game.state_from_board_full(bf2)
        fill_levels = game._get_fill_levels(state)
        opp_wins = {}
        opp_losses = {}
        other_moves = []
        opp_move_values = {}
        for col in possible_moves:
            assert state[fill_levels[col], col] == 0  # TODO: delete later
            state[fill_levels[col], col] = role  # Next state after move
            fill_levels[col] += 1
            value, moves = check_for_obvious_move(game, opponent, board_full=bf2,
                                                  state=state,
                                                  fill_levels=fill_levels,
                                                  depth=depth-1)
            opp_move_values[col] = value
            if value == win_value:
                opp_wins[col] = len(moves)
            elif value == loss_value:
                opp_losses[col] = len(moves)
            else:
                other_moves.append(col)
            # Reverse move
            fill_levels[col] -= 1
            state[fill_levels[col], col] = 0

        # 4. Take any move where opponent will definitely lose
        if len(opp_losses) > 0:
            return win_value, [col for col, value in opp_move_values.items()
                               if value == loss_value]

        # 5. If opponent will possibly win for all moves, assume defeat
        if len(opp_wins) == len(possible_moves):
            fewest = [col for col, n_win_moves in opp_wins.items()
                      if n_win_moves == min(opp_wins.values())]
            return loss_value, fewest

        # 6. Avoid any move where opponent will definitely win
        if len(opp_wins) > 0:
            return None, other_moves

    # Otherwise, return no value
    return None, list(possible_moves.keys())


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
    if state is None:
        state = game.state

    positions = []
    # Note: This is used to test different positions so it may not be role's
    # actual turn so role-checking is turned off
    for position in available_positions:
        next_state = game.next_state(state, (role, position), role_check=False)
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

    if state is None:
        state = game.state

    positions = []
    # Note: This is used to test different positions so it may not be role's
    # actual turn so role-checking is turned off
    for p1 in available_positions:
        next_state = game.next_state(state, (role, p1), role_check=False)
        remaining_positions = game.available_moves(state=next_state)
        p2s = []
        for p2 in remaining_positions:
            state2 = game.next_state(next_state, (role, p2), role_check=False)
            game_over, winner = game.check_game_state(state2, role)
            if winner == role:
                p2s.append(p2)
        if len(p2s) > 1:
            positions.append(p1)

    return positions


class Connect4BasicPlayer(Player):
    """Basic computer player that avoids obviously dumb moves.
    """

    def __init__(self, name="COMPUTER", seed=None, depth=3):
        super().__init__(name)
        self.depth = depth
        # Independent random number generator
        self.rng = random.Random(seed)

    def decide_next_move(self, game, role, show=False):

        move_format = game.help_text['Move format']

        value, moves = check_for_obvious_move(game, role, board_full=None,
                                       depth=self.depth)

        if len(game.moves) > 0:
            column = self.rng.choice(moves)
        else:
            column = 3
        move = (role, column)

        if show:
            print("%s's turn (%s): %s" % (self.name, move_format, str(column)))

        return move


# def test_player(player, game=Connect4Game, seed=1):
#     """
#     Calculates a score based on the player's performance playing 100
#     games of Connect4, 50 against a random player and 50 against
#     an expert. Score is calculated as follows:
#
#     score = (1 - random_player.games_won/50)* \
#             (random_player.games_lost/50)* \
#             (1 - expert_player.games_won/50)
#
#     An expert player should be able to get a score between 0.86 and
#     1.0 (it's not possible to always win against a random player).
#
#     Args:
#         player (Player): Player instance.
#         game (class): Class of game to use for the tests.
#         seed (int): Random number generator seed. Changing this
#             will change the test results slightly.
#
#     Returns:
#         score (float): Score between 0.0 and 1.0.
#     """
#
#     # Instantiate two computer opponents
#     random_player = RandomPlayer(seed=seed)
#     expert_player = Connect4Expert(seed=seed)
#
#     # Make a shuffled list of the order of play
#     opponents = [random_player]*50 + [expert_player]*50
#     rng = random.Random(seed)
#     rng.shuffle(opponents)
#
#     game = game()
#     player.updates_on, saved_mode = False, player.updates_on
#     for i, opponent in enumerate(opponents):
#         players = [player, opponent]
#         ctrl = GameController(game, players, move_first=i % 2)
#         ctrl.play(show=False)
#         game.reset()
#     player.updates_on = saved_mode
#
#     score = (1 - random_player.games_won / 50) * \
#             (random_player.games_lost / 50) * \
#             (1 - expert_player.games_won / 50)
#
#     return score
#
#
# def demo():
#     """Simple demo of TicTacToeGame game dynamics.
#     """
#
#     game = TicTacToeGame()
#     print("Game:", game)
#     print("Marks:", game.marks)
#     print("Roles:", game.roles)
#     print("State:\n", game.state)
#
#     print("Making some moves...")
#     game.make_move((1, (0, 2)), show=True)
#     game.make_move((2, (0, 1)), show=True)
#     game.make_move((1, (1, 1)), show=True)
#     game.make_move((2, (2, 2)), show=True)
#     game.show_state()
#     print("State:\n", game.state)
#
#     print("Game over:", game.game_over)
#
#     print("Moves so far:")
#     print(game.moves)
#
#     print("Turn:", game.turn)
#     print("Available moves:", game.available_moves())
#
#     game.make_move((1, (2, 0)), show=True)
#     game.show_state()
#
#     print("Game over:", game.game_over)
#     print("Winner:", game.winner)
#
#     game.reverse_move(show=True)
#     game.show_state()
#
#     print("Winner:", game.winner)
#
#     print("Try player 2 move...")
#     try:
#         game.make_move((2, (1, 2)))
#     except ValueError as err:
#         print(err)
#
#     print("Making some more moves...")
#     game.make_move((1, (1, 2)), show=True)
#     game.make_move((2, (2, 0)), show=True)
#     game.make_move((1, (0, 0)), show=True)
#     game.make_move((2, (1, 0)), show=True)
#     game.show_state()
#     game.make_move((1, (2, 1)), show=True)
#     print("Game over:", game.game_over)
#     print("Winner:", game.winner)
#
#
# def tictactoe_game(players, move_first=0, show=True):
#     """Demo of TicTacToeGame with two pre-defined players.
#
#     Args:
#         players (list): List of 2 Player instances.
#         move_first (int): Specify which player should go first.
#         show (bool): Print a message if True.
#     """
#
#     ctrl = GameController(TicTacToeGame(), players, move_first=move_first)
#     ctrl.play(show=show)
#
#
# def tictactoe_with_2_humans(names=("Player 1", "Player 2"), move_first=0,
#                             n=1):
#     """Demo of TicTacToeGame with two new human players.
#
#     Args:
#         names (list): A list containing two strings for names
#             of the players (optional).
#         move_first (int): Specify which player should go first.
#         n (int or None): Number of games to play.  If n=None,
#             it will loop indefinitely.
#     """
#
#     game = TicTacToeGame()
#     players = [HumanPlayer(name) for name in names]
#     play_looped_games(game, players, move_first=move_first, n=n)
#
#
# def main():
#     """Code to demonstrate use of this module."""
#
#     print("\nPlay Tic-Tac-Toe (Noughts and Crosses) against the "
#           "computer.")
#     game = TicTacToeGame()
#     computer_player = TDLearner("TD")
#     name = input("Enter your name: ")
#     human_player = HumanPlayer(name)
#     n_iterations = 1000
#
#     while True:
#
#         # Train computer against itself
#         # To do this you need to make a clone with the
#         # same value function
#         opponent = TDLearner("TD-clone")
#         opponent.value_function = computer_player.value_function
#
#         print("Computer is playing %d games against a clone of "
#               "itself..." % n_iterations)
#         train_computer_players(game, [computer_player, opponent],
#                                n_iterations)
#
#         print("Now play against it.")
#         game = TicTacToeGame()
#         players = [human_player, computer_player]
#         play_looped_games(game, players)
#
#         # Slowly reduce the learning rate
#         computer_player.learning_rate *= 0.9
#         computer_player.off_policy_rate *= 0.9
#
#         text = input("Press enter to do more training or 'q' to quit: ")
#         if text.strip().lower() == 'q':
#             break
#
#
# if __name__ == "__main__":
#     main()
