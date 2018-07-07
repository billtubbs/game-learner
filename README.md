# game-learner

Demonstration of a TD Learning algorithm learning to play the game [Tic Tac Toe (Noughts and Crosses)](https://en.wikipedia.org/wiki/Tic-tac-toe) based 
on the simple TD (temporal difference) algorithm described in Chapter 1 of the
[draft 2nd edition](www.incompleteideas.net/book/bookdraft2017nov5.pdf) of Sutton 
and Barto's book Reinforcement Learning: An Introduction.

Classes defined in `gamelearner.py`:

- `TicTacToeGame` - the game dynamics
- `HumanPlayer` - an interface to allow humans to play a game
- `TDLearner` - A simple TD learning algorithm that learns to play from experience
- `GameController` - controls a game between two players

Not implemented yet:
- `ExpertPlayer`

### Example usage

```
>>> from gamelearner import *
>>> game = TicTacToeGame()
>>> players = [HumanPlayer("Joe"), TDLearner("TD")]
>>> ctrl = GameController(game, players)
>>> ctrl.play(show=True)
_ _ _
_ _ _
_ _ _
TD's turn (row, col): (2, 1)
_ _ _
_ _ _
_ X _
Joe's turn (row, col): 2,0
_ _ _
_ _ _
O X _
TD's turn (row, col): (0, 2)
_ _ X
_ _ _
O X _
Joe's turn (row, col): 1,1
_ _ X
_ O _
O X _
TD's turn (row, col): (1, 2)
_ _ X
_ O X
O X _
Joe's turn (row, col): 2,2
_ _ X
_ O X
O X O
TD's turn (row, col): (1, 0)
_ _ X
X O X
O X O
Joe's turn (row, col): 0,0
Joe you won!
O _ X
X O X
O X O
Game over!
Joe won
```

