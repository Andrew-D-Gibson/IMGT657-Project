import random
import copy
import numpy as np

from TicTacToe import TicTacToe
from MCTS import MCTS

from Config import *


def simulateSelfPlayGames():
    training_examples = []

    for i in range(config['self_play']['num_of_self_play_games']):
        mcts = MCTS()

        while True:
            mcts.search()

            child_choice = np.random.choice(len(mcts.pi), p=mcts.pi)

            mcts = mcts.children[child_choice]

            game_over, value = mcts.board.is_game_over()
            if game_over:
                # Go back up a node to the last decision point
                mcts = mcts.parent

                # Make sure the value is set up for rewarding the right player
                if not mcts.board.player_1_move:
                    value = -value

                while True:
                    training_examples.append(mcts.board.get_training_example(mcts.pi) + (value,))
                    
                    if mcts.parent == None:
                        break

                    mcts = mcts.parent

                break     

    return trainingExamples

    

# Method for turning particular game states into training examples for keras models
def get_training_example(mcts):
    game_policy_size = mcts.board.get_keras_model().get_layer('policy_output').output_shape[1]


    legal_moves = mcts.board.get_legal_moves()
    assert(len(legal_moves) == len(mcts.pi))

    desired_output = np.zeros(game_policy_size)
    while len(mcts.pi) > 0:
        desired_output[legal_moves[0]] = mcts.pi[0]
        legal_moves.pop(0)
        mcts.pi.pop(0)

    return (mcts.board.get_keras_input(), desired_output)