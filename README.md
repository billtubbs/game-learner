# game-learner

Demonstration of a TD Learning algorithm learning to play the game [Tic Tac Toe (Noughts and Crosses)](https://en.wikipedia.org/wiki/Tic-tac-toe) based 
on the simple TD (temporal difference) algorithm described in Chapter 1 of the
[draft 2nd edition](www.incompleteideas.net/book/bookdraft2017nov5.pdf) of Sutton 
and Barto's book Reinforcement Learning: An Introduction.

Value update method:

V (s) ← V (s) + α􏰜[V (s′) − V (s)􏰝]

Classes defined in `gamelearner.py`:

- `TicTacToeGame` - the game dynamics
- `HumanPlayer` - an interface to allow humans to play a game
- `TDLearner` - A simple TD learning algorithm that learns to play from experience
- `GameController` - controls a game between two players
- `ExpertPlayer` - Computer algorithm to play optimally - should be unbeatable [according to wikipedia](https://en.wikipedia.org/wiki/Tic-tac-toe#Strategy)

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

Initially, the TDLearner makes random moves but gradually it updates its internal 
value function for each game state it visits until it (slowly) learns to make 
better moves, depending of course on how it's opponents play.

Running `gamelearner.py` instead of importing it will launch a looped game play
where 5 TD Learners play 1000 games against each other before the best of the 
five takes on a human.  After playing the human, the best player is cloned into
five new TD Learners which play another 1000 games against each other... and so
on.

```
$ python gamelearner.py

Play Tic-Tac-Toe (Noughts and Crosses) against 5
trained computer algorithms.
Enter your name: Joe

Training 5 computer players...
0 games completed
100 games completed
200 games completed
300 games completed
400 games completed
500 games completed
600 games completed
700 games completed
800 games completed
900 games completed

Results:
Draws: 94
TD02: 194
TD00: 195
TD04: 175
TD01: 164
TD03: 178
Size of value functions: [2490, 2669, 2581, 2481, 2378]
Best player so far: TDLearner('TD00')

Game of Tic Tac Toe with 2 players ['Joe', 'TD00']
_ _ _
_ _ _
_ _ _
Joe's turn (row, col): 
```

### Human-only play

If you want to play a game between two humans, call `game_with_2_humans()` as follows:

```
>>> from gamelearner import *
>>> game_with_2_humans(["Jack", "Jill"], move_first=0)

Game of Tic Tac Toe with 2 players ['Jack', 'Jill']
_ _ _
_ _ _
_ _ _
Jack's turn (row, col):
```

If `move_first` is not specified, the player to start the game is chosen randomly.
