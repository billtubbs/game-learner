#!/usr/bin/env python

"""Demonstration of reinforcement learning techniques that
learn to play simple games such as Tic-Tac-Tie (noughts and
crosses).
"""
# TODO:
# - Test the expert as TD seems to beat it sometimes
# - Implement save and load methods for TDLearners
# - Look for proven ways to reduce learning_rate
# - Allow alternating start player rather than random
# - Currently game.reset() will confuse TDLearner
#     Solution: Have TDLearner get previous move from game
# - create a game.make_moves method
# - On/off option for TD learning


import numpy as np
import itertools
import random
import datetime
from ast import literal_eval


class Player:
    all_players = {}

    def __init__(self, name):

        self.name = str(name)
        self.all_players[name] = self
        self.games_played = 0
        self.games_won = 0
        self.games_lost = 0

    def decide_next_move(self, game, role, show=False):
        """Override this method in your sub-class to return the
        player's next move in game assuming they are playing this
        role.

        Args:
            game (Game): Game which is being played
            role (object): Role that the player is playing (could
                           be int or str depending on game).
            show (bool): Print messages if True.

        Returns:
            move (tuple): Tuple containing (role, position).
        """

        raise NotImplementedError

        return move

    def make_move(self, game, role, show=False):

        assert not game.game_over, "Can't make move. Game is over."

        if game.moves is None:
            game.start()
        move = self.decide_next_move(game, role, show)
        game.make_move(move)

    def feedback(self, game, role, show=False):

        if game.game_over:
            self.games_played += 1
        if game.winner == role:
            self.games_won += 1
        if game.winner is not None:
            self.games_lost += 1

    def __repr__(self):

        return "Player(%s)" % self.name.__repr__()


class TicTacToeGame:
    """Simulates a game of tic tac toe (noughts and crosses)
    """

    name = 'Tic Tac Toe'
    size = (3, 3)
    roles = [1, 2]
    possible_n_players = [2]
    marks = ['X', 'O']

    help_text = {
        'move format': "row, col",
        'Move not available': "That position is not available.",
        'Number of players': "This game requires 2 players"
    }

    def __init__(self, moves=None):
        """Initialize a game.
        
        Args:
            moves (list): This is optional. Provide a list of completed
                moves. Each move should be a list or tuple of length 2
                where the first item is the player and the second is
                the board position (row, col).
        """

        self.n_players = 2
        self.winner = None
        self.game_over = False
        self.reset()
        if moves is not None:
            for move in moves:
                self.make_move(move)
            self.start()

    def start(self):
        """Record start time."""
        self.start_time = datetime.datetime.now()

    def stop(self):
        """Record end time."""
        self.stop_time = datetime.datetime.now()

    def reset(self):
        """Set the state of the game to the beginning (no moves).
        """

        self.moves = []
        self.player_iterator = itertools.cycle(self.roles)
        self.turn = next(self.player_iterator)
        self.state = np.zeros(self.size, dtype='b')
        self.winner = None
        self.game_over = False
        self.start_time = None
        self.end_time = None

    def show_state(self):
        """Display the current state of the board."""

        chars = '_' + ''.join(self.marks)
        for row in self.state:
            print(" ".join(list(chars[i] for i in row)))

    def available_positions(self, state=None):
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

    def update_state(self, state, move):
        """Updates the game state with the move to be taken.

        Args:
            state (np.ndarray): Array (size (3, 3)) of game state
            move (tuple): Tuple of length 2 containing the player role
                          and the move (role, position). Position is also a
                          tuple (row, col).

        Raises:
            ValueError if the position is out of bounds or if
            there is already a move in that position.

        Note: This method does not check if the move is valid.
        """

        role, position = move

        assert 0 <= position[0] < self.size[0]
        assert 0 <= position[1] < self.size[1]

        state[position] = role

    def next_state(self, move, state=None):
        """Returns the next state of the game if move were to be
        taken from current game state or from state parameter
        if provided.

        Args:
            move (tuple): Tuple of length 2 containing the player role
                          and the move (role, position). Position is also a
                          tuple (row, col)
            state (np.ndarray): Array (size (3, 3)) of game state or if
                                not provided the current game state will
                                be used.
        """

        if state is None:
            state = self.state.copy()
        next_state = state.copy()
        self.update_state(next_state, move)

        return next_state

    def make_move(self, move, show=False):
        """Update the game state with a new move.
        
        Args:
            move (tuple): Tuple of length 2 containing the player role
                          and the move (role, position). Position is also a
                          tuple (row, col).
            show (bool): Print a message if True.
        """

        assert self.winner is None, "Player %s has already won" % str(self.winner)

        role, position = move
        if self.turn != role:
            raise ValueError("It is not player %d's turn." % role)

        if self.state[position] != 0:
            raise ValueError(self.help_text['Move not available'])

        self.update_state(self.state, move)
        if show:
            print("Player %s made move %s" % (str(role), str(position)))
        self.moves.append(move)
        self.check_if_game_over()
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

    def check_game_state(self, state=None):
        """Check the game state provided to see whether someone
        has won or if it is draw.

        Args:
            state (np.array): If not None, check if this game state
                array is a game-over state, otherwise check the
                actual game state (self.state).

        returns:
            game_over, winner (bool, bool): If there is a winner,
                winner will be the winning role. If the game is over,
                game_over will be True.
        """

        game_over, winner = False, None

        if state is None:
            state = self.state

        for role in self.roles:
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

    def check_if_game_over(self):
        """Check to see whether someone has won or if it is draw. 
        If the game is over, game_over will be set to True.
        If there is a winner, the attribute winner will be set
        to the winning role.
        
        Returns:
            True if there is a winner else False.
        """

        self.game_over, self.winner = self.check_game_state()

        return self.game_over

    def __repr__(self):

        params = []
        if self.moves is not None:
            params.append("moves=%s" % self.moves.__repr__())

        return "TicTacToeGame(%s)" % ', '.join(params)


class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def decide_next_move(self, game, role, show=True):
        """Determine next move in the game game by getting input
        from the human player.

        Args:
            game (Game): Game which is being played
            role (object): Role that the player is playing (could
                           be int or str depending on game).
            show (bool): This has no effect. Messages are always
                         printed for human players.

        Returns:
            move (tuple): Tuple containing (role, position).
        """

        move_format = game.help_text['move format']
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
            if position in game.available_positions():
                break
            print(game.help_text['Move not available'])
            print("Try again.")

        return role, position

    def feedback(self, game, role, show=True):

        super().feedback(game, role)

        if game.game_over:
            if show:
                if game.winner == role:
                    print("%s you won!" % self.name)
                elif game.winner is not None:
                    print("%s you lost!" % self.name)

    def __repr__(self):

        return "HumanPlayer(%s))" % self.name.__repr__()


class TDLearner(Player):

    def __init__(self, name, learning_rate=0.5, off_policy_rate=0.1,
                 value_function=None):

        super().__init__(name)

        if value_function is None:
            value_function = {}
        self.value_function = value_function
        self.learning_rate = learning_rate
        self.off_policy_rate = off_policy_rate
        self.previous_states = dict()

    def decide_next_move(self, game, role, show=False):

        default_value = 1.0 / game.n_players
        move_format = game.help_text['move format']

        available_positions = game.available_positions()
        if len(available_positions) is 0:
            raise ValueError("There are no possible moves.")

        move_values = []
        for position in available_positions:
            next_state = game.next_state((role, position))
            key = generate_state_key(game, next_state, role)
            value = self.value_function.get(key, None)
            if value is None:
                value = default_value
                self.value_function[key] = value
            move_values.append((value, position, key))

        max_value = max(move_values)[0]
        best_moves = [move for move in move_values if move[0] == max_value]
        value, position, key = random.choice(best_moves)

        # Update value function
        if game in self.previous_states:
            previous_key = self.previous_states[game]
            self.value_function[previous_key] = (
                (1 - self.learning_rate) * self.value_function[previous_key] +
                self.learning_rate * value
            )

        self.previous_states[game] = key

        if show:
            print("%s's turn (%s): %s" % (self.name, move_format, str(position)))

        return role, position

    def feedback(self, game, role, show=False):

        super().feedback(game, role)

        if game.game_over:
            previous_key = self.previous_states[game]
            del self.previous_states[game]
            if game.winner == role:
                self.value_function[previous_key] = 1.0  # Reward for winning
            elif game.winner is None:
                self.value_function[previous_key] = 0.5  # Draw
                self.value_function[previous_key] = 0.5  # Draw
            else:
                self.value_function[previous_key] = 0.0  # Penalize for losing

    def copy(self, name):

        return TDLearner(name=name, learning_rate=self.learning_rate,
                         off_policy_rate=self.off_policy_rate,
                         value_function=self.value_function)

    def __repr__(self):

        return "TDLearner(%s)" % self.name.__repr__()


def generate_state_key(game, state, role):
    """Converts a game state in the form of an array into a
    string of bytes containing characters that represent the
    following:
     '-': Empty board position
     'S': Position occupied by self
     'O': Position occupied by opponent

    This is used by TDLearner to create unique hashable keys
    for storing values in a dictionary.

    Example (TicTacToeGame):
    > game.state
    array([[1, 0, 0],
           [2, 0, 0],
           [0, 0, 1]], dtype=int8)
    > players[1].generate_state_key(game, game.state, 1)
    b'S--O----S'

    Args:
        game (Game): Game that is being played
        state (np.ndarray): Game state array (shape may depend
                            on the game) of type int
        role (object): Role that the player is playing (could
                       be int or str depending on game).

    Returns:
        key (string): string of bytes representing game state.
    """

    # TODO: This only works for two-player games
    if role == game.roles[0]:
        chars = ['-', 'S', 'O']
    elif role == game.roles[1]:
        chars = ['-', 'O', 'S']
    else:
        raise NotImplementedError("Role not found for this game.")

    return np.array(chars, dtype='a')[state].tostring()


class ExpertPlayer(Player):

    def __init__(self, name):

        super().__init__(name)

    def winning_positions(self, game, role, available_positions, state=None):
        """Returns list of positions (row, col) that would result
        in player role winning if they took that position.

        Args:
            game (Game): Game that is being played
            role (object): Role that the player is playing (could
                           be int or str depending on game)
            available_positions (list): List of positions to search
            state (np.ndarray): Game state array (shape may depend
                                on the game) of type int

        Returns:
            positions (list):
        """

        positions = []
        for position in available_positions:
            next_state = game.next_state((role, position), state=state)
            game_over, winner = game.check_game_state(next_state)
            if winner == role:
                positions.append(position)

        return positions

    def fork_positions(self, game, role, available_positions, state=None):
        """Returns list of positions (row, col) where the player has
        two opportunities to win (two non-blocked lines of 2) if
        they took that position.

        Args:
            game (Game): Game that is being played
            role (object): Role that the player is playing (could
                           be int or str depending on game)
            available_positions (list): List of positions to search
            state (np.ndarray): Game state array (shape may depend
                                on the game) of type int

        Returns:
            positions (list):
        """

        positions = []
        for p1 in available_positions:
            next_state = game.next_state((role, p1), state=state)
            remaining_positions = game.available_positions(next_state)
            p2s = []
            for p2 in remaining_positions:
                state2 = game.next_state((role, p2), state=next_state)
                game_over, winner = game.check_game_state(state2)
                if winner == role:
                    p2s.append(p2)
            if len(p2s) > 1:
                positions.append(p1)

        return positions

    def decide_next_move(self, game, role, show=False):

        move_format = game.help_text['move format']
        available_positions = game.available_positions()
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        center = (1, 1)
        opponent = game.roles[(game.roles.index(game.turn) ^ 1)]

        move = None
        # 1. If first move of the game, play a corner or center
        if len(game.moves) == 0:
            move = (role, random.choice(corners + [center]))

        if move is None:
            # 2. Check for winning moves
            positions = self.winning_positions(game, role, available_positions)
            if positions:
                move = (role, random.choice(positions))

        if move is None:
            # 3. Check for blocking moves
            positions = self.winning_positions(game, opponent, available_positions)

            if positions:
                move = (role, random.choice(positions))

        if move is None:
            # 4. Check for fork positions
            positions = self.fork_positions(game, role, available_positions)
            if positions:
                move = (role, random.choice(positions))

        if move is None:
            # 5. Prevent opponent from using a fork position
            opponent_forks = self.fork_positions(game, opponent, available_positions)
            positions = []
            for p1 in available_positions:
                next_state = game.next_state((role, p1))
                p2s = self.winning_positions(game, role, available_positions, next_state)
                for p2 in p2s:
                    state2 = game.next_state((role, p2), state=next_state)
                    game_over, winner = game.check_game_state(state2)
                    if winner == role and p2 not in opponent_forks:
                        positions.append(p1)
            if positions:
                move = (role, random.choice(positions))

        if move is None:
            # 6. Try to play center
            if center in available_positions:
                move = (role, (1, 1))

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
            positions = [corner for corner in corners if game.state[corner] == 0]
            if positions:
                move = (role, random.choice(positions))

        if move is None:
            # 9. Play anywhere else - i.e. a middle position on a side
            move = (role, random.choice(available_positions))

        if show:
            print("%s's turn (%s): %s" % (self.name, move_format, str(move)))

        return move

    def __repr__(self):

        return "ExpertPlayer(%s)" % self.name.__repr__()


class GameController:
    def __init__(self, game, players, move_first=None):

        self.game = game
        assert len(players) in game.possible_n_players, \
            game.help_text['Number of players']
        self.players = players
        if move_first is None:
            move_first = random.randint(0, 1)
        player_roles = game.roles if move_first == 0 else \
            list(reversed(game.roles))
        self.player_roles = dict(zip(self.players, player_roles))
        self.players_by_role = dict(zip(player_roles, self.players))

    def announce_game(self):

        items = (
            self.game.name,
            len(self.players),
            str([p.name for p in self.players])
        )
        print("\nGame of %s with %d players %s" % items)

    def play(self, n=None, show=True):
        """Play the game until game.game_over is True or after
        n_moves if n_moves is > 0.

        Args:
            n (int): Number of moves to play (optional).
            show (bool): Print messages if True.
        """

        assert self.game.game_over is not True, "Game is over. Use game.reset() to play again."

        if self.game.start_time is None:
            self.game.start()
        while not self.game.game_over:
            if show:
                self.game.show_state()
            player = self.players_by_role[self.game.turn]
            player.make_move(self.game, self.player_roles[player], show=show)
            if n is not None:
                n -= 1
                if n < 1:
                    break

        for player in self.players:
            player.feedback(self.game, self.player_roles[player], show=show)
        if show:
            self.announce_result()

    def announce_result(self):

        self.game.show_state()
        if self.game.game_over:
            print("Game over!")
            if self.game.winner:
                print("%s won" % self.players_by_role[self.game.winner].name)
            else:
                print("Draw")

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

    print("Game over:", game.check_if_game_over())

    print("Moves so far:")
    game.show_moves()

    print("Turn:", game.turn)
    print("Available moves:", game.available_positions())

    game.make_move((1, (2, 0)))
    game.show_state()

    print("Game over:", game.check_if_game_over())
    print("Winner:", game.winner)

    game.reverse_move(show=True)
    game.show_state()

    print("Winner:", game.winner)

    try:
        game.make_move((2, (1, 2)))
    except ValueError:
        print("Player 2 tried to go when it wasn't their turn")

    print("Making some more moves...")
    game.make_move((1, (1, 2)), show=True)
    game.make_move((2, (2, 0)), show=True)
    game.make_move((1, (0, 0)), show=True)
    game.make_move((2, (1, 0)), show=True)
    game.show_state()
    game.make_move((1, (2, 1)), show=True)
    print("Game over:", game.check_if_game_over())
    print("Winner:", game.winner)


def game_with_2_humans(names=("Player 1", "Player 2"), move_first=None):
    """Demo of TicTacToeGame with two new human players.

    Args:
        names (list): A list containing two strings for names
                      of the players (optional)
        move_first (int): Specify which player should go first.
                          Random if not specified.
    """

    players = [HumanPlayer(name) for name in names]
    ctrl = GameController(TicTacToeGame(), players, move_first=move_first)
    ctrl.announce_game()
    ctrl.play(show=True)


def game_with_2_players(players, move_first=None, show=True):
    """Demo of TicTacToeGame with two pre-defined players.

    Args:
        players (list): List of 2 Player instances
        move_first (int): Specify which player should go first.
                          Random if not specified.
        show (bool): Print a message if True.
    """

    assert len(players) == 2
    ctrl = GameController(TicTacToeGame(), players, move_first=move_first)
    ctrl.announce_game()
    ctrl.play(show=show)


def train_computer_players(players, iterations=1000):
    """Play repeated games with n computer players then
    play against one of them.

    Args:
        players (list): List of 2 Player instances
        iterations (int): Number of iterations of training.
    """

    n_players = len(TicTacToeGame.roles)
    stats = {}

    print("\nTraining %d computer players..." % len(players))
    for i in range(iterations):
        game = TicTacToeGame()
        selected_players = random.sample(players, n_players)
        ctrl = GameController(game, selected_players)
        ctrl.play(show=False)
        if game.winner:
            player = ctrl.players_by_role[game.winner]
            # print("Player %s won" % str(player.name))
            stats[player] = stats.get(player, 0) + 1
        else:
            # print("Draw")
            stats["Draws"] = stats.get("Draws", 0) + 1
        if i % 100 == 0:
            print(i, "games completed")

    print("\nResults:")
    for p, count in stats.items():
        print("%s: %d" % (p if p == "Draws" else p.name, count))


def looped_games(players):
    """Play repeated games between two players.

    Args:
        players (list): List of 2 Player instances.
    """

    while True:
        game = TicTacToeGame()
        ctrl = GameController(game, players)
        ctrl.announce_game()
        ctrl.play(show=True)
        text = input("Press enter to play again or s to stop: ")
        if text.lower() == 's':
            break

    print("Results:")
    for player in players:
        items = (player.name, player.games_won, player.games_played)
        print("Player %s won %d of %d games" % items)


def main():
    n = 5
    print("\nPlay Tic-Tac-Toe (Noughts and Crosses) against %d\n"
          "trained computer algorithms." % n)
    computer_players = [TDLearner("TD%02d" % i) for i in range(n)]

    name = input("Enter your name: ")
    human_player = HumanPlayer(name)

    while True:
        train_computer_players(computer_players, 1000)

        best_wins = max([p.games_won for p in computer_players])
        best_players = [p for p in computer_players if p.games_won == best_wins]
        best_player = random.choice(best_players)
        print("Best player so far:", best_player)

        looped_games([human_player, best_player])
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
