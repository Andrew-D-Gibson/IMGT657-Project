import numpy as np

from MCTS import MCTS

from Config import *

def head_to_head_game(mcts_1, mcts_2, stochastic = True):
    # Reset the game state in case these players are playing multiple games in a row
    mcts_1.reset()
    mcts_2.reset()

    # This loop always searches with player "1", and switches between the two players till the game is over
    # Note that the value returned is always 1 for a player 1 win, so we don't have to worry about admin
    while True:
        game_over, value = mcts_1.board.is_game_over()
        if game_over:
            return value
        
        mcts_1.search()
        mcts_2.expand_node()

        if stochastic:
            child_choice = np.random.choice(len(mcts_1.pi), p=mcts_1.pi)
        else:
            child_choice = np.argmax(mcts_1.pi)

        mcts_1 = mcts_1.children[child_choice]
        mcts_2 = mcts_2.children[child_choice]

        # Flip players for the next search
        mcts_1, mcts_2 = mcts_2, mcts_1


def head_to_head_match(mcts_1, mcts_2, 
                       num_of_games = config['self_play']['num_of_testing_games'], 
                       stochastic = True):
    mcts_1_wins = 0
    mcts_2_wins = 0
    draws = 0

    for i in range(num_of_games):
        print(f"{i+1}/{config['self_play']['num_of_testing_games']}, ", end='')
        result = head_to_head_game(mcts_1, mcts_2, stochastic)

        # Flip results as we flip players
        if i % 2 == 1:
            result *= -1

        # Tally the result properly
        if result == 1:
            mcts_1_wins += 1
        elif result == -1:
            mcts_2_wins += 1
        else:
            draws += 1

        # Flip players
        mcts_1, mcts_2 = mcts_2, mcts_1

    return mcts_1_wins, draws, mcts_2_wins

            