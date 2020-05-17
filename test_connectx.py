# !/usr/bin/env python
"""Unit Tests for connectx.py.  Run this script to check
everything is working.
"""

import unittest
import numpy as np
from numpy.testing import assert_array_equal

from connectx import Connect4, wins_from_next_move
from gamelearner import RandomPlayer, GameController
#from gamelearner import train_computer_players


class TestConnectX(unittest.TestCase):

    def test_check_game_state_after_move(self):

        # Setup a board state
        moves = [
            (1, 3), (2, 6), (1, 0), (2, 5), (1, 2),
            (2, 2), (1, 5), (2, 1), (1, 3), (2, 1),
            (1, 6), (2, 2), (1, 4), (2, 4), (1, 4),
            (2, 4), (1, 1)
        ]
        game = Connect4(moves=moves)
        directions = ['u', 'd', 'r', 'l', 'ur', 'dr', 'ul', 'dl']
        self.assertEqual(list(game._steps.keys()), directions)

        # Prepare a move by player 1
        role = 1
        column = 3
        move = (role, column)
        level = game._fill_levels[column]
        position = (level+1, column+1)
        self.assertEqual(game._board_full[position], 0)

        # Calculate chains in each direction from the move position
        chain_lengths = {
            direction: game._chain_in_direction(position, direction, role)
            for direction in directions
        }
        self.assertEqual(
            chain_lengths,
            {'u': 0, 'd': 2, 'r': 1, 'l': 0, 'ur': 0, 'dr': 0, 'ul': 0, 'dl': 0}
        )
        self.assertFalse(game._check_game_state_after_move(move))

        # Prepare a move by player 2
        role = 2
        move = (role, column)
        # Calculate chains in each direction from the move position
        chain_lengths = {
            direction: game._chain_in_direction(position, direction, role)
            for direction in directions
        }
        self.assertEqual(
            chain_lengths,
            {'u': 0, 'd': 0, 'r': 0, 'l': 1, 'ur': 1, 'dr': 2, 'ul': 0, 'dl': 2}
        )
        self.assertTrue(game._check_game_state_after_move(move))

        # Test other board states
        board_full, state = game._empty_board_state()
        state[:] = 1
        role = 1
        position = (1, 1)
        chain_lengths = {
            direction: game._chain_in_direction(position, direction, role, board_full=board_full)
            for direction in directions
        }
        self.assertEqual(
            chain_lengths,
            {'u': 3, 'd': 0, 'r': 3, 'l': 0, 'ur': 3, 'dr': 0, 'ul': 0, 'dl': 0}
        )

        # Loop along diagonal
        chain_lengths_1 = {d: [] for d in directions}
        chain_lengths_2 = {d: [] for d in directions}
        for x in range(6):
            position = (x+1, x+1)  # Position (x, x) in board_full array
            for direction in game._steps.keys():
                length_1 = game._chain_in_direction(position, direction, 1, board_full=board_full)
                length_2 = game._chain_in_direction(position, direction, 2, board_full=board_full)
                chain_lengths_1[direction].append(length_1)
                chain_lengths_2[direction].append(length_2)
        self.assertEqual(
            chain_lengths_1,
            {
                'u': [3, 3, 3, 2, 1, 0],
                'd': [0, 1, 2, 3, 3, 3],
                'r': [3, 3, 3, 3, 2, 1],
                'l': [0, 1, 2, 3, 3, 3],
                'ur': [3, 3, 3, 2, 1, 0],
                'dr': [0, 1, 2, 3, 2, 1],
                'ul': [0, 1, 2, 2, 1, 0],
                'dl': [0, 1, 2, 3, 3, 3]
            }
        )
        self.assertEqual(
            chain_lengths_2,
            {
                'u': [0, 0, 0, 0, 0, 0],
                'd': [0, 0, 0, 0, 0, 0],
                'r': [0, 0, 0, 0, 0, 0],
                'l': [0, 0, 0, 0, 0, 0],
                'ur': [0, 0, 0, 0, 0, 0],
                'dr': [0, 0, 0, 0, 0, 0],
                'ul': [0, 0, 0, 0, 0, 0],
                'dl': [0, 0, 0, 0, 0, 0]
            }
        )

    def test_check_positions(self):
        
        game = Connect4()
        state = np.array([
            [2, 1, 2, 1, 0, 2, 2],
            [1, 2, 1, 2, 2, 1, 0],
            [1, 2, 1, 1, 0, 0, 0],
            [2, 0, 1, 1, 1, 2, 0],
            [2, 0, 0, 0, 0, 0, 1],
            [0, 2, 1, 1, 0, 0, 0]
        ])
        positions = (state == 1).astype('int8')
        connect = game._check_positions(positions)
        self.assertTrue(connect)
        positions = (state == 2).astype('int8')
        connect = game._check_positions(positions)
        self.assertFalse(connect)
    
    def test_available_moves(self):

        game = Connect4()
        state = game.state.copy()

        state[:] = np.zeros(game.shape, dtype='int8')
        self.assertEqual(
            game.available_moves(state).tolist(),
            [0, 1, 2, 3, 4, 5, 6]
        )
        state[0, 3] = 1
        self.assertEqual(
            game.available_moves(state).tolist(),
            [0, 1, 2, 3, 4, 5, 6]
        )
        state[0:5, 1] = 2
        self.assertEqual(
            game.available_moves(state).tolist(),
            [0, 1, 2, 3, 4, 5, 6]
        )
        state[0:6, game.shape[1]-1] = 1
        self.assertEqual(
            game.available_moves(state).tolist(),
            [0, 1, 2, 3, 4, 5]
        )
        state[:, :] = 1
        self.assertEqual(
            game.available_moves(state).tolist(),
            []
        )

    def test_check_game_state(self):

        game = Connect4(moves=None)
        state = np.zeros(game.shape)
        assert game.check_game_state(state) == (False, None)

        state = np.array([[2, 2, 1, 2, 0, 2, 1],
                [1, 0, 0, 0, 2, 2, 0],
                [1, 2, 1, 2, 0, 1, 1],
                [0, 2, 1, 2, 1, 1, 1],
                [0, 2, 0, 2, 1, 0, 1],
                [2, 1, 0, 2, 2, 0, 2]])
        assert game.check_game_state(state) == (True, 2)

        state = np.array([
            [0, 0, 2, 0, 2, 2, 2],
            [1, 1, 2, 1, 1, 2, 0],
            [1, 1, 1, 1, 2, 0, 2],
            [0, 1, 2, 0, 0, 2, 0],
            [2, 1, 2, 0, 0, 2, 0],
            [0, 1, 2, 2, 1, 2, 0]
        ])
        assert game.check_game_state(state) == (True, 1)

        state = np.array([
            [2, 0, 2, 1, 0, 2, 1],
            [0, 1, 1, 0, 1, 2, 0],
            [1, 0, 0, 0, 2, 0, 2],
            [2, 1, 0, 0, 1, 0, 2],
            [0, 2, 1, 0, 1, 2, 0],
            [0, 0, 1, 2, 1, 0, 1]
        ])
        assert game.check_game_state(state) == (False, None)

        state = np.array([
            [0, 1, 0, 2, 2, 1, 1],
            [2, 1, 1, 1, 0, 1, 2],
            [2, 2, 0, 1, 1, 1, 0],
            [1, 2, 1, 0, 0, 2, 1],
            [1, 2, 2, 2, 1, 0, 1],
            [0, 0, 2, 2, 1, 0, 1]
        ])
        assert game.check_game_state(state) == (True, 2)

        state = np.array([
            [2, 2, 2, 1, 2, 2, 2],
            [1, 1, 1, 2, 1, 1, 1],
            [2, 2, 2, 1, 2, 2, 2],
            [1, 1, 1, 2, 1, 1, 1],
            [2, 2, 2, 1, 2, 2, 2],
            [1, 1, 1, 2, 1, 1, 1]
        ])  # draw
        assert game.check_game_state(state) == (True, None)

    def test_game_execution(self):
        """Steps through a game of Connect 4 checking
        various attributes and methods.
        """

        game = Connect4()
        self.assertEqual(game.name, 'Connect 4')
        self.assertEqual(game.connect, 4)
        self.assertEqual(game.roles, [1, 2])
        self.assertEqual(game.shape, (6, 7))
        self.assertEqual(game.possible_n_players, [2])
        self.assertEqual(game.marks, ['X', 'O'])

        assert_array_equal(game.state, np.zeros((6, 7), dtype='int8'))
        self.assertEqual(game._board_full.shape, (8, 9))
        self.assertTrue(np.all(game._fill_levels == 0))
        self.assertFalse(game.game_over)
        self.assertEqual(game.winner, None)

        game.make_move((1, 0))

        # Check game attributes
        assert_array_equal(game._fill_levels,
                           np.array([1, 0, 0, 0, 0, 0, 0]))
        self.assertFalse(game.game_over)
        self.assertEqual(game.winner, None)
        self.assertEqual(game.get_rewards(), {2: 0.0})

        game.make_move((2, 1))

        assert_array_equal(game._fill_levels,
                           np.array([1, 1, 0, 0, 0, 0, 0]))
        self.assertFalse(game.game_over)
        self.assertEqual(game.winner, None)
        self.assertEqual(game.get_rewards(), {1: 0.0})

        game.make_move((1, 2))
        game.make_move((2, 0))

        # Check state
        test_state = np.array([
            [1, 2, 1, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ], dtype='int8')
        assert_array_equal(game.state, test_state)

        test_board_full = np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1,  1,  2,  1,  0,  0,  0,  0, -1],
            [-1,  2,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0,  0,  0,  0,  0,  0, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        ], dtype='int8')
        assert_array_equal(game._board_full, test_board_full)

        test_fill_levels = np.array([2, 1, 1, 0, 0, 0, 0], dtype='int8')
        assert_array_equal(game._fill_levels, test_fill_levels)

        test_pos_last = (1, 0)
        self.assertEqual(game._pos_last, test_pos_last)

        self.assertFalse(game.game_over)
        self.assertEqual(game.winner, None)

        moves = [(1, 0), (2, 1), (1, 2), (2, 0)]
        self.assertEqual(game.moves, moves)

        self.assertEqual(game.turn, 1)
        assert_array_equal(
            np.array(game.available_moves()), 
            np.array([0, 1, 2, 3, 4, 5, 6])
        )

        # Initialize game from same state
        game = Connect4(moves=moves)
        assert_array_equal(game.state, test_state)
        self.assertFalse(game.game_over)
        self.assertEqual(game.winner, None)
        assert_array_equal(game._fill_levels, test_fill_levels)

        game.make_move((1, 6))

        assert_array_equal(
            game.state, 
            np.array([
                [1, 2, 1, 0, 0, 0, 1],
                [2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ], dtype='int8')
        )
        self.assertEqual(game.check_game_state(), (False, None))
        self.assertEqual(game.turn, 2)

        # Test reversing move
        game.reverse_move()
        assert_array_equal(game.state, test_state)
        self.assertEqual(game.turn, 1)
        assert_array_equal(game._board_full, test_board_full)
        assert_array_equal(game._fill_levels, test_fill_levels)

        # Test new game with existing moves
        moves = [
            (1, 0), (2, 1), (1, 0), (2, 1),
            (1, 0), (2, 1)
        ]
        game = Connect4(moves=moves)
        assert_array_equal(game.moves, moves)
        self.assertEqual(game.check_game_state(), (False, None))
        game.make_move((1, 0))
        self.assertEqual(game.check_game_state(), (True, 1))
        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, 1)

        # Check terminal rewards
        rewards = game.get_terminal_rewards()
        self.assertEqual(
            rewards, {1: game.terminal_rewards['win'],
                      2: game.terminal_rewards['lose']}
        )

        self.assertEqual(game._pos_last, (3, 0))

        game.reverse_move()
        assert_array_equal(
            game.moves,
            [
                (1, 0), (2, 1), (1, 0), (2, 1),
                (1, 0), (2, 1)
            ]
        )

        assert_array_equal(
            game.state,
            np.array([
                [1, 2, 0, 0, 0, 0, 0],
                [1, 2, 0, 0, 0, 0, 0],
                [1, 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ])
        )
        
        # self.assertTrue(np.array_equal(game.state, state))
        self.assertTrue(game.winner is None)

        with self.assertRaises(ValueError) as context:
            game.make_move((2, 1))
        self.assertTrue("It is not player 2's turn." in str(context.exception))

        # Make more moves until game ends
        while not game.game_over:
            role = game.turn
            moves = game.available_moves()
            move = np.random.choice(moves)
            game.make_move((role, move))

        rewards = game.get_terminal_rewards()
        if game.winner is None:
            self.assertEqual(len(game.available_moves()), 0)
            self.assertEqual(rewards, {1: game.terminal_rewards['draw'],
                                       2: game.terminal_rewards['draw']})
        elif game.winner == 1:
            self.assertEqual(rewards, {1: game.terminal_rewards['win'],
                                       2: game.terminal_rewards['lose']})
        elif game.winner == 2:
            self.assertEqual(rewards, {1: game.terminal_rewards['lose'],
                                       2: game.terminal_rewards['win']})

        # Test a game that ends in a draw
        moves = [
            (1, 2), (2, 3), (1, 4), (2, 1), (1, 1), (2, 5),
            (1, 2), (2, 1), (1, 4), (2, 5), (1, 1), (2, 0),
            (1, 0), (2, 4), (1, 1), (2, 3), (1, 1), (2, 2),
            (1, 0), (2, 4), (1, 0), (2, 5), (1, 5), (2, 6),
            (1, 6), (2, 2), (1, 3), (2, 0), (1, 3), (2, 4),
            (1, 3), (2, 3), (1, 6), (2, 5), (1, 4), (2, 2),
            (1, 2), (2, 0), (1, 5), (2, 6), (1, 6), (2, 6)
        ]
        self.assertEqual(len(moves), game.shape[0]*game.shape[1])
        game = Connect4()
        for move in moves:
            game.make_move(move)
        self.assertEqual(len(game.moves), len(moves))
        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, None)

    def test_with_GameController(self):

        game = Connect4()
        player1 = RandomPlayer('R1')
        player2 = RandomPlayer('R2')

        players = [player1, player2]
        ctrl = GameController(game, players)
        ctrl.play(show=False)
        self.assertTrue(game.game_over)

    # def test_generate_state_key(self):
    #     """Test generate_state_key method of TicTacToeGame.
    #     """

    #     game = TicTacToeGame()
    #     game.state[:] = [[1, 0, 0],
    #                      [2, 0, 0],
    #                      [0, 0, 1]]
    #     self.assertEqual(
    #         game.generate_state_key(game.state, 1), b'S--O----S'
    #     )
    #     self.assertEqual(
    #         game.generate_state_key(game.state, 2), b'O--S----O'
    #     )

    # def test_with_players(self):

    #     game = TicTacToeGame()
    #     players = [RandomPlayer(seed=1), RandomPlayer(seed=1)]
    #     ctrl = GameController(game, players)
    #     ctrl.play(show=False)
    #     final_state = np.array([
    #         [1, 2, 1],
    #         [2, 1, 1],
    #         [1, 2, 2]
    #     ])
    #     self.assertTrue(np.array_equal(ctrl.game.state, final_state))
    #     self.assertEqual(game.game_over, 1)
    #     self.assertEqual(game.winner, 1)

    # def test_expert_player(self):

    #     results = []
    #     game = TicTacToeGame()
    #     expert_player1 = TicTacToeExpert("EXP1", seed=1)
    #     expert_player2 = TicTacToeExpert("EXP2", seed=1)
    #     random_player = RandomPlayer(seed=1)
    #     players = [expert_player1, expert_player2, random_player]
    #     game_stats = train_computer_players(game, players, iterations=100,
    #                                         seed=1, show=False)
    #     self.assertTrue(game_stats[expert_player1]['lost'] == 0)
    #     self.assertTrue(game_stats[expert_player2]['lost'] == 0)
    #     self.assertTrue(game_stats[random_player]['won'] == 0)

    #     # Save results
    #     results.append({player.name: stat for player, stat in
    #                     game_stats.items()})

    #     # Check repeatability with random seed set
    #     game.reset()
    #     expert_player1 = TicTacToeExpert("EXP1", seed=1)
    #     expert_player2 = TicTacToeExpert("EXP2", seed=1)
    #     random_player = RandomPlayer(seed=1)
    #     players = [expert_player1, expert_player2, random_player]
    #     game_stats = train_computer_players(game, players, iterations=100,
    #                                         seed=1, show=False)
    #     self.assertTrue(game_stats[expert_player1]['lost'] == 0)
    #     self.assertTrue(game_stats[expert_player2]['lost'] == 0)
    #     self.assertTrue(game_stats[random_player]['won'] == 0)

    #     # Save results
    #     results.append({player.name: stat for player, stat in
    #                     game_stats.items()})

    #     self.assertTrue(results[0] == results[1])

class TestAnalysisFunctions(unittest.TestCase):

    def test_wins_from_next_move(self):

        # Get empty board state
        game = Connect4()
        board_full, state = game._empty_board_state()

        role = 1
        self.assertEqual(
            list(wins_from_next_move(game, role, board_full=board_full).keys()),
            list(range(7))
        )
        self.assertFalse(
            any(wins_from_next_move(game, role, board_full=board_full).values())
        )

        # First move by 1
        state[0, 3] = 1
        role = 2
        self.assertFalse(
            any(wins_from_next_move(game, role, board_full=board_full).values())
        )

        # Board state
        state[:] = np.array([
            [0, 1, 1, 1, 2, 2, 1],
            [0, 0, 1, 0, 2, 1, 0],
            [0, 0, 0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 1, 0]
        ])
        role = 1
        self.assertEqual(
            wins_from_next_move(game, role, board_full=board_full),
            {0: True, 1: False, 2: False, 3: False, 4: False, 6: False}
        )
        role = 2
        self.assertEqual(
            wins_from_next_move(game, role, board_full=board_full),
            {0: False, 1: False, 2: False, 3: False, 4: True, 6: False}
        )

        state[:] = np.array([
            [0, 1, 2, 1, 2, 2, 1],
            [0, 0, 1, 0, 2, 1, 0],
            [0, 0, 0, 0, 2, 1, 0],
            [0, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 1, 0]
        ])
        role = 1
        self.assertEqual(
            wins_from_next_move(game, role, board_full=board_full),
            {0: False, 1: False, 2: False, 3: False, 4: False, 6: False}
        )
        role = 2
        self.assertEqual(
            wins_from_next_move(game, role, board_full=board_full),
            {0: False, 1: False, 2: False, 3: True, 4: True, 6: False}
        )

        state[:] = np.array([
            [0, 0, 1, 1, 2, 2, 1],
            [0, 0, 1, 2, 2, 1, 0],
            [0, 0, 0, 0, 2, 2, 0],
            [0, 0, 0, 0, 1, 2, 0],
            [0, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 1, 0]
        ])
        role = 1
        self.assertEqual(
            wins_from_next_move(game, role, board_full=board_full),
            {0: False, 1: False, 2: False, 3: False, 4: False, 6: False}
        )
        role = 2
        self.assertEqual(
            wins_from_next_move(game, role, board_full=board_full),
            {0: False, 1: False, 2: False, 3: False, 4: False, 6: False}
        )

        state[:] = np.array([
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ])
        role = 2
        self.assertEqual(
            wins_from_next_move(game, role, board_full=board_full),
            {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False}
        )


if __name__ == '__main__':
    unittest.main()
