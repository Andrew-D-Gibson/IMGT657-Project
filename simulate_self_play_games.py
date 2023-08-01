import random
import copy
import numpy as np

from TicTacToe import TicTacToe
from MCTS import MCTS

from Config import *


# Method for turning particular game states into training examples for keras models
def get_training_example(mcts):
    game_policy_size = mcts.board.policy_size

    legal_moves = mcts.board.get_legal_moves()
    assert(len(legal_moves) == len(mcts.pi))

    desired_output = np.zeros(game_policy_size)
    while len(mcts.pi) > 0:
        desired_output[legal_moves[0]] = mcts.pi[0]
        
        legal_moves.pop(0)
        mcts.pi.pop(0)

    return (mcts.board.get_keras_input(), desired_output)


# Method for simulating a number of games
# Returns the games as individual training examples, unshuffled
def simulate_self_play_games(keras_model = None):
    training_examples = []

    for i in range(config['self_play']['num_of_self_play_games']):
        if keras_model == None:
            mcts = MCTS()
        else:
            mcts = MCTS(keras_model=keras_model)

        while True:
            mcts.search()

            # Pick the next node using the search probabilities
            child_choice = np.random.choice(len(mcts.pi), p=mcts.pi)

            # Actually move to the next node
            mcts = mcts.children[child_choice]

            # Check if the game is over, and if so create all the training examples
            game_over, value = mcts.board.is_game_over()
            if game_over:
                # Game over means there are no decisions to be made/learned from, 
                # so go back up a node to the last decision point
                mcts = mcts.parent

                while True:
                    training_examples.append(get_training_example(mcts) + (value,))
                    
                    if mcts.parent == None:
                        break

                    mcts = mcts.parent

                break     

    return training_examples