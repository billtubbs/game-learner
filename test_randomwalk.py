# !/usr/bin/env python
"""Unit Tests for randomwalk.py.  Run this script to check
everything is working.
"""

import unittest
import numpy as np

from randomwalk import calculate_true_values, rms_error, RandomWalkGame


class TestRandomWalkGame(unittest.TestCase):

    def test_calculate_true_values(self):
        """Test function for calculating true state-values.
        """

        size = 19
        game = RandomWalkGame(size=size,
                              terminal_rewards={'T1': -1.0, 'T2': 1.0})
        calculated_true_values = calculate_true_values(game)
        true_values = np.array([
            -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,  0.,
            0.1, 0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9
        ])
        self.assertTrue(np.isclose(calculated_true_values, true_values).all())

    def test_rms_error(self):
        size = 19
        game = RandomWalkGame(size=size,
                              terminal_rewards={'T1': -1.0, 'T2': 1.0})
        true_values = calculate_true_values(game)
        values = np.zeros(size)
        self.assertEqual(rms_error(values, true_values), 0.5477225575051662)

        values = np.ones(size)*3
        rms_error(values, true_values)
        self.assertEqual(rms_error(values, true_values), 3.049590136395381)


if __name__ == '__main__':
    unittest.main()
