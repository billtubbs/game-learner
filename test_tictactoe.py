# !/usr/bin/env python
"""Unit Tests for tictactoe.py.  Run this script to check
everything is working.
"""

import unittest
import numpy as np

from tictactoe import TicTacToeGame

class TestTicTacToe(unittest.TestCase):

    def test_check_game_state(self):
        """Test methods for checking game state (they should
        be identical).
        """

        game = TicTacToeGame()
        for i in range(5):
            if i == 0:
                state = np.zeros((3, 3), dtype=int)
            else:
                state = np.random.randint(0, 3, size=9).reshape((3, 3))
            result1 = game.check_game_state(state)  # Numpy indexing method
            result2 = game.check_game_state(state, calc=True)  # Numpy.sum method
            self.assertEqual(result1, result2), "Error in TicTacToeGame.check_game_state"

    def test_game_execution(self):
        """Steps through one game of Tic-Tac-Toe checking
        various attributes and methods.
        """

        game = TicTacToeGame()
        self.assertEqual(game.roles, [1, 2])
        self.assertTrue(
            np.array_equal(game.state, np.zeros((game.size, game.size)))
        )

        # Make some moves
        game.make_move((1, (0, 2)))
        game.make_move((2, (0, 1)))
        game.make_move((1, (1, 1)))
        game.make_move((2, (2, 2)))

        self.assertFalse(game.game_over)

        state = np.array([
            [0, 2, 1],
            [0, 1, 0],
            [0, 0, 2]
        ])

        self.assertTrue(
            np.array_equal(game.state, state)
        )

        self.assertEqual(
            game.moves, [(1, (0, 2)), (2, (0, 1)),
                         (1, (1, 1)), (2, (2, 2))]
        )

        self.assertEqual(game.turn, 1)
        self.assertEqual(
            game.available_moves(), [(0, 0), (1, 0), (1, 2),
                                     (2, 0), (2, 1)]
        )

        game.make_move((1, (2, 0)))
        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, 1)

        game.reverse_move()
        self.assertTrue(np.array_equal(game.state, state))
        self.assertTrue(game.winner is None)

        with self.assertRaises(Exception) as context:
            game.make_move((2, (1, 2)))
        self.assertTrue("It is not player 2's turn." in str(context.exception))

        # Make some more moves...
        game.make_move((1, (1, 2)))
        game.make_move((2, (2, 0)))
        game.make_move((1, (0, 0)))
        game.make_move((2, (1, 0)))
        self.assertEqual(game.state[2, 1], 0)

        game.make_move((1, (2, 1)))
        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, None)

    def test_generate_state_key(self):
        """Test generate_state_key method of TicTacToeGame.
        """

        game = TicTacToeGame()
        game.state[:] = [[1, 0, 0],
                         [2, 0, 0],
                         [0, 0, 1]]
        self.assertEqual(
            game.generate_state_key(game.state, 1), b'S--O----S'
        )
        self.assertEqual(
            game.generate_state_key(game.state, 2), b'O--S----O'
        )


if __name__ == '__main__':
    unittest.main()