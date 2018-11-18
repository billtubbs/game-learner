# game-learner

Demonstration of a TD Learning algorithm learning to play the game [Tic Tac Toe (Noughts and Crosses)](https://en.wikipedia.org/wiki/Tic-tac-toe) based 
on the simple one-step TD (temporal difference) algorithm described in Chapter 1 of the
[draft 2nd edition](www.incompleteideas.net/book/bookdraft2017nov5.pdf) of Sutton 
and Barto's book Reinforcement Learning: An Introduction.

Value function update method:

V(s) ← V(s) + α[reward + γV(s′) − V(s)]

Classes defined in `gamelearner.py`:

- `TicTacToeGame` - the game dynamics
- `HumanPlayer` - an interface to allow humans to play a game
- `TDLearner` - a simple TD learning algorithm that learns to play from experience
- `ExpertPlayer` - computer algorithm to play optimally - should be unbeatable [according to wikipedia](https://en.wikipedia.org/wiki/Tic-tac-toe#Strategy)
- `RandomPlayer` - computer player that makes random moves
- `GameController` - controls a game between two players

### Example usage

```
>>> from gamelearner import *
>>> game = TicTacToeGame()
>>> players = [HumanPlayer("Joe"), TDLearner("TD")]
>>> ctrl = GameController(game, players)
>>> ctrl.play()
Game of Tic Tac Toe with 2 players ['Joe', 'TD']
_ _ _
_ _ _
_ _ _
Joe's turn (row, col): 0,0
X _ _
_ _ _
_ _ _
TD's turn (row, col): (0, 1)
X O _
_ _ _
_ _ _
Joe's turn (row, col): 1,1
X O _
_ X _
_ _ _
TD's turn (row, col): (2, 1)
X O _
_ X _
_ O _
Joe's turn (row, col): 2,2
Joe you won!
X O _
_ X _
_ O X
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

Play Tic-Tac-Toe (Noughts and Crosses) against the computer.
Enter your name: Joe
Computer is playing 1000 games against a clone of itself...

Training 2 computer players...
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
TD: won 432, lost 443, drew 125
TD-clone: won 443, lost 432, drew 125
Now play against it.

Game of Tic Tac Toe with 2 players ['Joe', 'TD']
_ _ _
_ _ _
_ _ _
Joe's turn (row, col): 
```

### Training with an expert

To use the expert player to train a TD Learner player you can use this function:

```
>>> from gamelearner import *
>>> computer_players = [TDLearner("TD1"), ExpertPlayer("EXPERT")]
>>> train_computer_players(computer_players)

Training 2 computer players...
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
TD1: won 0, lost 613, drew 387
EXPERT: won 613, lost 0, drew 387
>>> train_computer_players(computer_players)

Training 2 computer players...
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
TD1: won 0, lost 230, drew 770
EXPERT: won 230, lost 0, drew 770
```

### Performance metric

There is a also a test metric which tests an algorithm's performance playing
games against an expert player and a player that makes random moves.

```
>>> from gamelearner import *
>>> td1, td2 = [TDLearner("TD%d" % i) for i in range(2)]
>>> random_player = RandomPlayer()
>>> expert_player = ExpertPlayer()
>>> players = [td1, td2, random_player, expert_player]
>>> for i in range(10):
...     train_computer_players(players, 500, show=False)
...     td1_score = test_player(td1)
...     print("Score after %d games: %5.2f" % (td1.games_played, td1_score))
... 
Score after 258 games:  0.02
Score after 521 games:  0.04
Score after 757 games:  0.06
Score after 997 games:  0.11
Score after 1240 games:  0.15
Score after 1494 games:  0.18
Score after 1738 games:  0.20
Score after 1978 games:  0.48
Score after 2234 games:  0.29
Score after 2479 games:  0.35
>>> test_player(expert_player)
0.98
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
