from gamelearner import *
from tictactoe import TicTacToeGame, TicTacToeExpert, test_player

seed = 1
td1, td2 = TDLearner("TD1"), TDLearner("TD2", seed=seed),
expert_player = TicTacToeExpert("Expert", seed=seed)
random_player = RandomPlayer("Random", seed=seed)

players = [td1, td2, random_player, expert_player]

game = TicTacToeGame()

train_computer_players(game, players, 1000, seed=seed)

scores = {player.name: round(test_player(player), 2) for player in players}

print("Scores:", scores)

assert expert_player.games_lost == 0