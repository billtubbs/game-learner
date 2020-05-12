#!/usr/bin/env python
"""Demonstration of TD reinforcement learning algorithms
described in Chapters 1 and 6 of the 2nd edition of Sutton and
Barto's book Reinforcement Learning: An Introduction.
"""

# TODO list:
# - Separate value estimation (prediction) from behaviour:
#     Need to restructure players to separate policies from
#     agent class and potentially have two policies.
# - Add a role attribute to game.available_moves
# - Merge terminal_update() with update()
# - Add timestep index to TDLearner's saved states and do checks
# - Replace use of 'position' with 'action'
# - game.next_state is basically the environment dynamics. Should
#     re-design so you can use it independent of self.
# - TDLearner uses after-states but should be able to use state-action
#     pairs as well.
# - Add other methods - n-step TD, monte-carlo, DP
# - Add other methods - Sarsa
# - create a game.make_moves method
# - Are there proven ways to reduce learning_rate?
# - Allow player to be initialised from pickle file
# - Consider using property decorators
# - Consider renaming 'position' as 'action' for generality

from abc import ABC, abstractmethod
import numpy as np
from collections import deque
import random
import datetime
import pickle
from ast import literal_eval


class Player(ABC):
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

    @abstractmethod
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

        move = None

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
        """Override this method if player uses rewards to learn
        how to play.  Note, if the game has ended (game.gameover
        == True) then reward must be a terminal reward which may
        come in addition to a regular reward for the previous
        move made.

        Args:
            game (Game): Game that player is playing.
            reward (float): Reward value based on the last move
                made by player.
            show (bool): Print a message (optional).
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
        ride this method, make sure to call super().gameover(game, role)
        so that the player keeps a record of games played, won,
        lost.

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

        return f"{self.__class__.__name__}({self.name.__repr__()})"


class Environment(ABC):
    """Simulate a game environment.

    Class attributes:
        Environment.name (str): The environment's name (e.g. 'Tic Tac Toe').
        Environment.roles (list): Possible player roles (e.g. [1, 2]).
        Environment.possible_n_players (list) : List of integers specifying
            possible number of players.  E.g. for Tic-Tac-Toe,
            possible_n_players = [2].
        Environment.help_text (dict): Help messages for human players.
    """

    name = 'Abstract environment'
    roles = None
    possible_n_players = None
    help_text = None

    def __init__(self, start_state, moves=None):
        """Initialize the environment.

        Args:
            moves (list): Optional. Provide a list of completed moves.
                Each move should be a list or tuple of length 2
                where the first item is the player role and the second is
                the action taken.
        """

        self.start_state = start_state
        self.state = start_state
        self.start_time = None
        self.end_time = None
        self.game_over = False
        self.winner = None  # TODO: Decide if this needs to be in abstract class
        self.moves = []
        if moves is not None:
            for move in moves:
                self.make_move(move)
            self.check_if_game_over()

    def start(self):
        """Records start time (self.start_time)."""

        self.start_time = datetime.datetime.now()

    def stop(self):
        """Records end time (self.end_time)."""

        self.end_time = datetime.datetime.now()

    def reset(self):
        """Sets the state of the game back to the beginning
        (no moves made).
        """

        self.start_time = None
        self.end_time = None
        self.game_over = False
        self.moves = []

    @abstractmethod
    def show_state(self):
        """Displays the current state of the environment."""

        pass

    @abstractmethod
    def available_moves(self, state=None):
        """Returns list of possible actions in current state.

        Args:
            state (...): Environment state.  If not provided,
                the current environment state will be used.
        """

        # TODO: In general, this should have a role attribute

        pass

    @abstractmethod
    def next_state(self, state, move):
        """Returns the next state of the environment if move
        were to be taken from current state or from state if
        provided.

        Args:
            state (..): State of environment
            move (tuple): Tuple of length 2 containing a
                player role and move.

        Returns:
            next_state (..): Copy of state after move made.

        Raises:
            ValueError if the move is not possible.
        """

        pass

    def update_state(self, move):
        """Updates the environment state with the move to be
        taken.

        Args:
            move (tuple): Tuple of length 2 containing a player
                role and move (role, move).
        """

        self.state = self.next_state(self.state, move)

    def make_move(self, move, show=False):
        """Update the game state with a new move.

        Args:
            move (tuple): Tuple of length 2 containing a
                player role and action (role, action).
            show (bool): Print a message if True.
        """

        role, action = move
        assert role in self.roles, f"{role.__repr__()} is not a " \
                                   "valid player role."

        assert self.game_over is False, "Game is already over."

        self.update_state(move)
        self.moves.append(move)

        if show:
            print(f"Player {role} made move {action.__repr__()}")

        self.check_if_game_over(role)
        if self.game_over:
            self.stop()

    @abstractmethod
    def reverse_move(self, show=False):
        """Reverse the last move made.

        Args:
            show (bool): Print a message if True.
        TODO: Better to store previous states and revert back rather
        # than create a 'previous_state' method or something
        # Or simply remove this feature - why do we need it?
        """

        pass

    @abstractmethod
    def get_rewards(self):
        """Returns any rewards at the current time step for
        players.
        """

        pass

    @abstractmethod
    def get_terminal_rewards(self):
        """Returns the rewards at the end of the game for both
        players.
        """

        assert self.game_over

        pass

    @abstractmethod
    def check_game_state(self, state=None, role=None):
        """Check the environment state to see if episode
        will terminate now.

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

        pass

    def check_if_game_over(self, role=None):
        """Checks game state using self.check_game_state.  If the
        game is over, game_over will be set to True.  If there
        is a winner, the attribute winner will be set to the
        winning role. This method is automatically called by make_move
        and reverse_move and when a game with existing moves is
        initialized.

        Args:
            role (int): If specified, only check for a win by this
            game role.

        Returns:
            True if there is a winner else False.
        """

        self.game_over, self.winner = self.check_game_state(role=role)

    @abstractmethod
    def generate_state_key(self, state, role):
        """Converts a game state (or afterstate) into a hashable key
        that can be used by tabular value functions for storing and
        retrieving state values.

        Implement this method for each environment.

        Could be a string of bytes or an integer for example (as long
        as there is a unique correspondence between states and keys).

        Returns:
            key (string or int): Unique key representing a game state.
        """

        pass

    def __repr__(self):

        params = []
        if self.moves:
            params.append(f"moves={self.moves.__repr__()}")

        return f"{self.__class__.__name__}({', '.join(params)})"


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


class TDLearner(Player):
    def __init__(self, name="TD", learning_rate=0.1, gamma=1.0,
                 off_policy_rate=0.1, initial_value=0.5,
                 value_function=None, use_afterstates=True,
                 seed=None):
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
            seed (int): Random number generator seed value.
        """

        super().__init__(name)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.off_policy_rate = off_policy_rate
        self.initial_value = initial_value
        if value_function is None:
            value_function = {}
        self.value_function = value_function
        self.saved_game_states = {}
        self.on_policy = None

        # Dedicated random number generator for sole use
        self.seed = seed
        self.rng = random.Random(seed)

    def get_value(self, state_key):
        """Returns a value from TDLearner's value_function for the
        game state represented by state_key. If there is no item for
        that state, returns the initial_value instead.
        """

        return self.value_function.get(state_key, self.initial_value)

    def save_state(self, game, state_key):
        """Adds state_key to a list of keys stored in dictionary
        self.saved_game_states for each game being played.
        """

        self.saved_game_states.get(game, []).append(state_key)

    def decide_next_move(self, game, role, show=False):

        available_positions = game.available_moves()
        if len(available_positions) == 0:
            raise ValueError("There are no possible moves.")

        elif self.rng.random() < self.off_policy_rate:
            # Random off-policy move
            self.on_policy = False
            position = self.rng.choice(available_positions)
            next_state = game.next_state(game.state, (role, position))
            next_state_key = game.generate_state_key(next_state, role)

        else:
            # On-policy move
            self.on_policy = True

            # Uses 'after-state' values
            options = []
            for position in available_positions:
                next_state = game.next_state(game.state, (role, position))
                next_state_key = game.generate_state_key(next_state, role)
                action_value = self.get_value(next_state_key)
                options.append((action_value, position, next_state_key))

            max_value = max(options)[0]
            best_options = [m for m in options if m[0] == max_value]
            _, position, next_state_key = self.rng.choice(best_options)

        # Save chosen state for learning updates later
        self.save_state(game, next_state_key)

        if show:
            move_format = game.help_text['Move format']
            print("%s's turn (%s): %s" % (self.name, move_format,
                                          str(position)))

        return role, position

    def get_saved_game_states(self, game):
        """Returns the list self.saved_game_states.  If it
        doesn't exists, assigns an empty list to it first.
        """

        states = self.saved_game_states.get(game, None)
        if states is None:
            states = []
            self.saved_game_states[game] = states

        return states

    def update(self, game, reward, show=False):
        """Update TDLearner's value function based on current reward
        from game.  This gets called by GameController during a game
        whenever rewards are distributed and the player has received
        one.

        Args:
            game (Game): Game that player is playing.
            reward (float): Reward value.
            show (bool): Print a message if True.
        """

        if self.updates_on and self.on_policy is True:

            # Retrieve previous game-states if there were any
            states = self.get_saved_game_states(game)

            if not game.game_over:

                # Need at least 2 previous actions for a value update
                if len(states) > 1:

                    # TD value function update
                    self.value_function[states[-2]] = \
                        self.get_value(states[-2]) + self.learning_rate*(
                            reward + self.gamma*self.get_value(states[-1]) -
                            self.get_value(states[-2])
                        )
            else:
                # Reward must be a terminal state reward
                # Update previous state-value if there was one
                if states:
                    last_state = states[-1]

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

    def copy(self, name=None):

        if name is None:
            name = self.name

        return TDLearner(name=name,
                         learning_rate=self.learning_rate,
                         gamma=self.gamma,
                         off_policy_rate=self.off_policy_rate,
                         initial_value=self.initial_value,
                         value_function=self.value_function.copy(),
                         use_afterstates=self.use_afterstates,
                         seed=self.seed)


class RandomPlayer(Player):
    def __init__(self, name="RANDOM", seed=None):
        """Tic-Tac-Toe game player that makes random moves.

        Args:
            name (str): Arbitrary name to identify the player
            seed (int): Random number generator seed value.
        """

        # TODO: Need to restructure players to separate policy from agent
        super().__init__(name)

        # Dedicated random number generator
        self.seed = seed
        self.rng = random.Random(seed)

    def decide_next_move(self, game, role, show=False):
        available_moves = game.available_moves()
        position = self.rng.choice(available_moves)
        move = (role, position)

        if show:
            move_format = game.help_text['Move format']
            print("%s's turn (%s): %s" % (self.name, move_format, str(position)))

        return move


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
                self.players_by_role[role].update(self.game, reward,
                                                  show=show)

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

        params = [self.game.__repr__(), self.players.__repr__()]

        return f"{self.__class__.__name__}({', '.join(params)})"


def train_computer_players(game, players, iterations=1000, seed=None,
                           show=True):
    """Play repeated games with n computer players then play
    against one of them.

    Args:
        game (Environment): Game environment to play.
        players (list): List of at least 2 Player instances.
        iterations (int): Number of iterations of training.
        seed (int): Random number generator seed value.
        show (bool): Print progress messages and results if True.

    returns:
        stats (dict): Dictionary containing the game results for
            each player.
    """

    n_players = game.possible_n_players[0]
    assert len(players) >= n_players, "Provide at least 2 players to train."

    stats = {p: {'won': 0, 'lost': 0, 'played': 0} for p in players}

    # Dedicated random-number generator
    rng = random.Random(seed)

    if show:
        print("\nTraining %d computer players..." % len(players))
    for i in range(iterations):
        game.reset()
        selected_players = rng.sample(players, n_players)
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

    return stats


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
            text = input("Press enter to play again or 's' to stop: ")
            if text.strip().lower() == 's':
                break

        ctrl.reset()

    print("\nResults:")
    wins = 0
    for player in players:
        items = (player.name, player.games_won, player.games_played)
        print("Player %s won %d of %d games" % items)
        wins += player.games_won
