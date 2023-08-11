import random
import copy
import os
import numpy as np
import tensorflow as tf
import ray

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
@ray.remote
def simulate_self_play_games():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    training_examples = []

    mcts = MCTS(keras_model=tf.keras.models.load_model('Networks/Best_Model'))

    for i in range(config['self_play']['num_of_self_play_games']):
        print(f"{i+1}/{config['self_play']['num_of_self_play_games']}, ", end='')
        if ((i+1)%10==0):
            print(' ')

        mcts.reset()

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