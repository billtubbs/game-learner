#!/usr/bin/env python

"""Demonstration of reinforcement learning techniques that
learn to play simple games such as Tic-Tac-Tie (noughts and
crosses).
"""
# TODO:
# - Create an expert player for testing/training
# - Implement save and load methods for TDLearners
# - Look for proven ways to reduce learning_rate
# - Allow alternative start player rather than random
# - Currently game.reset() will confuse TDLearner.
#   Solution: Have TDLearner get previous move from game.


import numpy as np
import itertools
import random
import datetime


class Player:
    all_players = {}

    def __init__(self, name):

        self.name = str(name)
        self.all_players[name] = self
        self.games_played = 0
        self.games_won = 0

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

        if game.moves is None:
            game.start()
        move = self.decide_next_move(game, role, show)
        game.make_move(move)

    def feedback(self, game, role, show=False):

        if game.game_over:
            self.games_played += 1
        if game.winner == role:
            self.games_won += 1

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
        'Move not available': "That position is already taken.",
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
        self.reset()
        if moves is not None:
            for move in moves:
                self.make_move(move)
            self.start()
        self.winner = None

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

    def available_positions(self):
        """Returns list of available (empty) board positions (x, y).
        """

        x, y = np.where(self.state == 0)
        return list(zip(x, y))

    def update_state(self, state, move):
        """Updates the game state with the move to be taken.

        Args:
            state (np.ndarray): Array (size (3, 3)) of game state
            move (tuple): Tuple of length 2 containing the player role
                          and the move (role, position). Position is also a
                          tuple (row, col).
        """

        role, position = move

        assert 0 <= position[0] < self.size[0]
        assert 0 <= position[1] < self.size[1]
        assert self.winner is None, "Player %s has already won" % str(self.winner)

        if state[position] != 0:
            raise ValueError("There is already a move in that position.")

        if self.turn != role:
            raise ValueError("It is not player %d's turn." % role)

        state[position] = self.turn

    def next_state(self, move):
        """Returns the future state of the game if move were to
        be taken.

        Args:
            move (tuple): Tuple of length 2 containing the player role
                          and the move (role, position). Position is also a
                          tuple (row, col).
        """

        next_state = self.state.copy()
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

        self.update_state(self.state, move)
        if show:
            print("Player %s made move %s" % (str(move[0]), str(move[1])))
        self.moves.append(move)
        self.check_if_game_over()
        if self.game_over:
            self.stop()
        self.turn = next(self.player_iterator)

    def reverse_move(self, show=False):
        """Reverse the last move made.
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

    def check_if_game_over(self):
        """Check to see whether someone has won or if it is draw. 
        If there is a winner, the attribute winner will be set 
        to the winning player. If the game is over, game_over will
        be set to True.
        
        Returns:
            True if there is a winner else False.
        """

        self.winner = None
        self.game_over = False

        for role in self.roles:
            moves = (self.state == role)
            if any((
                    np.any(moves.sum(axis=0) == 3),
                    np.any(moves.sum(axis=1) == 3),
                    (np.diagonal(moves).sum() == 3),
                    (np.diagonal(np.fliplr(moves)).sum() == 3)
            )):
                self.winner = role
                self.game_over = True

        if np.sum(self.state == 0) == 0:
            self.game_over = True

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
                position = eval(text)
            except (SyntaxError, NameError):
                print("Move format is %s" % move_format)
                continue
            if position in game.available_positions():
                break
            else:
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
        self.previous_state = None

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
        if self.previous_state is not None:
            self.value_function[self.previous_state] = (
                (1 - self.learning_rate) * self.value_function[self.previous_state] +
                self.learning_rate * value
            )
        if show:
            print("%s's turn (%s): %s" % (self.name, move_format, str(position)))

        self.previous_state = key

        return role, position

    def feedback(self, game, role, show=False):

        super().feedback(game, role)

        if game.game_over:
            if game.winner == role:
                key = generate_state_key(game, game.state, role)
                self.value_function[key] = 1.0
            else:
                self.value_function[self.previous_state] = 0.0

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
    def __init__(self, name, show=False):

        super().__init__(name, show)

    def decide_next_move(self, game, role, show=False):

        move_format = game.help_text['move format']
        possible_moves = {role: [] for role in game.roles}
        for role in game.roles:
            player_moves = (game.state == role)
            a0 = player_moves.sum(axis=0) == 2
            if np.any(a0):
                pass
                # for i in np.where(a0 == True)[0].tolist():
                #    np.where(a[i] == True)[0].tolist()
                #    possible_moves[role].append((i,))
            a1 = np.any(player_moves.sum(axis=1) == 2)
            d0 = np.diagonal(player_moves).sum() == 2
            d1 = np.diagonal(np.fliplr(player_moves)).sum() == 2
            move = NotImplementedError

        if show:
            print("%s's turn (%s): %s" % (self.name, move_format, str(move)))

        return move

    def __repr__(self):

        return "TDLearner(%s, %d)" % (self.name, self.n_players)


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

    def play(self, n=None, show=False):
        """Play the game until game.check_if_game_over is True
        or after n_moves if n_moves is > 0.

        Args:
            n (int): Number of moves to play (optional).
            show (bool): Print messages if True.
        """

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
            self.game.show_state()
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


def game_with_2_existing_players(players, move_first=None):
    """Demo of TicTacToeGame with two pre-defined players.

    Args:
        players (list): List of 2 Player instances
        move_first (int): Specify which player should go first.
                          Random if not specified.
    """

    assert len(players) == 2
    ctrl = GameController(TicTacToeGame(), players, move_first=move_first)
    ctrl.announce_game()
    ctrl.play(show=True)


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

    print("Size of value functions:",
          [len(p.value_function) for p in players]
          )


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
