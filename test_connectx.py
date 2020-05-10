# !/usr/bin/env python
"""Unit Tests for connectx.py.  Run this script to check
everything is working.
"""

import unittest
import numpy as np
from numpy.testing import assert_array_equal

from connectx import Connect4
from gamelearner import train_computer_players


class TestConnectX(unittest.TestCase):

    def test_game_execution(self):
        """Steps through a game of Connect 4 checking
        various attributes and methods.
        """

        game = Connect4()
        self.assertEqual(game.roles, [1, 2])
        self.assertEqual(game.shape, (6, 7))
        
        self.assertTrue(
            np.array_equal(game.state, np.zeros(game.shape))
        )
        assert_array_equal(game.state , np.zeros((6, 7), dtype='int8'))
        self.assertEqual(game._board_full.shape, (8, 9))

        # Make some moves
        game.make_move((1, 0))

        # Check rewards
        self.assertEqual(game.get_rewards(), {2: 0.0})

        game.make_move((2, 1))

        self.assertEqual(game.get_rewards(), {1: 0.0})

        game.make_move((1, 2))
        game.make_move((2, 0))

        # Check state
        test_state = np.array(
            [[1, 2, 1, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]], dtype='int8'
        )
        assert_array_equal(game.state, test_state)

        self.assertFalse(game.game_over)

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

        game.make_move((1, 6))

        assert_array_equal(
            game.state, 
            np.array(
                [[1, 2, 1, 0, 0, 0, 1],
                [2, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]], dtype='int8')
        )
        self.assertEqual(game.check_game_state(), (False, None))

        moves = [
            (1, 0), (2, 1), (1, 0), (2, 1),
            (1, 0), (2, 1)
        ]
        game = Connect4(moves=moves)
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

        # game.reverse_move()
        # self.assertTrue(np.array_equal(game.state, state))
        # self.assertTrue(game.winner is None)

        # with self.assertRaises(Exception) as context:
        #     game.make_move((2, (1, 2)))
        # self.assertTrue("It is not player 2's turn." in str(context.exception))

        # # Make some more moves...
        # game.make_move((1, (1, 2)))
        # game.make_move((2, (2, 0)))
        # game.make_move((1, (0, 0)))
        # game.make_move((2, (1, 0)))
        # self.assertEqual(game.state[2, 1], 0)

        # game.make_move((1, (2, 1)))
        # self.assertTrue(game.game_over)
        # self.assertEqual(game.winner, None)
        # rewards = game.get_terminal_rewards()
        # self.assertEqual(
        #     rewards, {1: game.terminal_rewards['draw'],
        #               2: game.terminal_rewards['draw']}
        # )

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


if __name__ == '__main__':
    unittest.main()
