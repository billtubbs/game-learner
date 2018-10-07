# game-learner

Demonstration of a TD Learning algorithm learning to play the game [Tic Tac Toe (Noughts and Crosses)](https://en.wikipedia.org/wiki/Tic-tac-toe) based 
on the simple one-step TD (temporal difference) algorithm described in Chapter 1 of the
[draft 2nd edition](www.incompleteideas.net/book/bookdraft2017nov5.pdf) of Sutton 
and Barto's book Reinforcement Learning: An Introduction.

Value function update method:

V (s) ← V (s) + α[V (s′) − V (s)]

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
TD00: won 185, lost 183
TD01: won 168, lost 163
TD02: won 180, lost 168
TD03: won 176, lost 187
TD04: won 168, lost 176
Draws: 42
Best player so far: TDLearner('TD00')

Game of Tic Tac Toe with 2 players ['Joe', 'TD00']
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
TD1: won 0, lost 338, drew 662
EXPERT: won 338, lost 0, drew 662
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
TD1: won 0, lost 160, drew 840
EXPERT: won 160, lost 0, drew 840
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
...     train_computer_players(players, 100, show=False)
...     td1_score = test_player(td1)
...     print("Score after %d games: %5.2f" % (td1.games_played, td1_score))
... 
Score after 44 games:  0.03
Score after 94 games:  0.05
Score after 144 games:  0.11
Score after 185 games:  0.08
Score after 238 games:  0.09
Score after 288 games:  0.09
Score after 342 games:  0.19
Score after 382 games:  0.13
Score after 429 games:  0.15
Score after 475 games:  0.08
>>> test_player(expert_player)
0.96
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
