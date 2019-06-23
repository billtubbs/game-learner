#!/usr/bin/env python
"""Demonstration of TD reinforcement learning algorithms
described in Chapters 1 and 6 of the 2nd edition of Sutton and
Barto's book Reinforcement Learning: An Introduction.

Algorithm learns to play the Tic-Tac-Tie (noughts and crosses)
game. It can be trained against itself or against an expert
algorithm.
"""

import numpy as np
import itertools
from gamelearner import *


class TicTacToeGame(Environment):
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
    terminal_rewards = {'win': 1.0, 'lose': 0.0, 'draw': 0.5}

    help_text = {
        'Move format': "row, col",
        'Move not available': "That position is not available.",
        'Number of players': "This game requires 2 players.",
        'Out of range': "Row and column must be in range 0 to %d." % (size - 1)
    }

    # These are array indices used by check_game_state method
    row_idxs = np.array([
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (0, 1, 2),
        (0, 1, 2),
        (0, 1, 2),
        (0, 1, 2),
        (0, 1, 2),
    ])

    col_idxs = np.array([
        (0, 1, 2),
        (0, 1, 2),
        (0, 1, 2),
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (0, 1, 2),
        (2, 1, 0)
    ])

    def __init__(self, moves=None):
        """Initialize a game.

        Args:
            moves (list): This is optional. Provide a list of completed
                moves. Each move should be a list or tuple of length 2
                where the first item is the player role and the second is
                the board position (row, col).
        """

        start_state = np.zeros(self.shape, dtype='b')
        super().__init__(start_state, moves=moves)
        self.n_players = 2
        # self.winner = None  TODO: Currently this is done in parent class
        self.player_iterator = itertools.cycle(self.roles)
        self.turn = next(self.player_iterator)
        self.start_state = start_state


    def reset(self):
        """Set the state of the game back to the beginning
        (no moves made).
        """

        super().reset()
        self.player_iterator = itertools.cycle(self.roles)
        self.turn = next(self.player_iterator)
        self.state = self.start_state
        self.winner = None

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

    def next_state(self, state, move, role_check=True):
        """Returns the next state of the game if move were to be
        taken from current game state or from state if provided.

        Args:
            state (np.ndarray): Array (size (3, 3)) of game state or if
                not provided the current game state will be used.
            move (tuple): Tuple of length 2 containing the player role
                and the move (role, position). Position is also a tuple
                (row, col).
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

        assert 0 <= position[0] < self.size, self.help_text['Out of range']
        assert 0 <= position[1] < self.size, self.help_text['Out of range']
        assert state[position] == 0, self.help_text['Move not available']

        next_state = state.copy()
        next_state[position] = role

        return next_state

    def update_state(self, move):
        """Updates the game state with the move to be taken.

        Args:
            move (tuple): Tuple of length 2 containing the player role
                and the move (role, position). Position is also a tuple
                (row, col).
        """

        self.state = self.next_state(self.state, move)

    def make_move(self, move, show=False):
        """Update the game state with a new move.

        Args:
            move (tuple): Tuple of length 2 containing a
                player role and action (role, action).
            show (bool): Print a message if True.
        """

        super().make_move(move, show)
        self.turn = next(self.player_iterator)

    def reverse_move(self, show=False):
        """Reverse the last move made.

        Args:
            show (bool): Print a message if True.
        TODO: This method may not need overloading. See gamelearner.py.
        """

        last_move = self.moves.pop()
        self.state[last_move[1]] = 0  # Removes last move from board
        self.turn = next(self.player_iterator)  # TODO: Only works for 2 player games!
        self.check_if_game_over()

    def get_rewards(self):
        """Returns any rewards at the current time step for
        players.  In TicTacToe, there are no rewards until the
        end of the game so it sends a zero reward to each
        player after their opponent has made their move.
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

    def check_game_state(self, state=None, role=None, calc=False):
        """Check the game state provided to see whether someone
        has won or if it is draw.

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

        game_over, winner = False, None

        if state is None:
            state = self.state

        # If role specified, only check for a win by role
        if role:
            roles = [role]
        else:
            roles = self.roles

        if calc is False:
            # Check for a win state using previously prepared
            # array indices
            lines = state[self.row_idxs, self.col_idxs]
            for role in roles:
                if ((lines == role).sum(axis=1) == 3).any():
                    game_over, winner = True, role
                    break

        else:
            # This alternative method checks for a win using numpy
            # methods - reliable but not as fast as method above
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

        if winner is None and np.all(state > 0):
            game_over = True

        return game_over, winner

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

        # TODO: Instead of a char array should this simply be an integer?
        # (Would probably conserve memory).  Could provide hash function
        # so states are still decodable.
        if role == self.roles[0]:
            chars = ['-', 'S', 'O']
        elif role == self.roles[1]:
            chars = ['-', 'O', 'S']
        else:
            raise ValueError("Role does not exist in this game.")

        return np.array(chars, dtype='a')[state].tostring()


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


class TicTacToeExpert(Player):
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
                    next_state = game.next_state(game.state, (role, p1))
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
    expert_player = TicTacToeExpert(seed=seed)

    # Make a shuffled list of the order of play
    opponents = [random_player]*50 + [expert_player]*50
    rng = random.Random(seed)
    rng.shuffle(opponents)

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


def tictactoe_with_2_humans(names=("Player 1", "Player 2"), move_first=0,
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


def main():
    """Code to demonstrate use of this module."""

    print("\nPlay Tic-Tac-Toe (Noughts and Crosses) against the "
          "computer.")
    game = TicTacToeGame()
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
        train_computer_players(game, [computer_player, opponent],
                               n_iterations)

        print("Now play against it.")
        game = TicTacToeGame()
        players = [human_player, computer_player]
        play_looped_games(game, players)

        # Slowly reduce the learning rate
        computer_player.learning_rate *= 0.9
        computer_player.off_policy_rate *= 0.9

        text = input("Press enter to do more training or 'q' to quit: ")
        if text.strip().lower() == 'q':
            break


if __name__ == "__main__":
    main()
