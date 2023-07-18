import copy
import numpy as np

import TTTBoard

class trainingExample:
    def __init__(self, board, search_probs):
        # TEMP
        self.x_move = copy.deepcopy(board.x_move)
        
        # Copy the board values, and flip who's to play if necessary
        self.board_array = board.getArrayRepresentation()
  
        # Copy the search probabilities, re-assigning them to match their position in the board
        self.search_probs = np.zeros(81)
        raw_search_probs = np.array(search_probs)
        possible_moves = board.findMoves()
        for i in range(81):
            if i in possible_moves:
                try:
                    self.search_probs[i] = raw_search_probs[0]
                    raw_search_probs = np.delete(raw_search_probs,0)
                except:
                    print("Unknown error.")
                    board.print()
                    print(search_probs)
                    print(possible_moves)
                    print(raw_search_probs)
                    print(raw_search_probs.shape)
                    print(i)
                    print(self.search_probs)
                    continue
            else:
                continue
        
    def addReward(self, reward):
        if self.x_move:
            self.reward = np.array([reward])
        else:
            self.reward = np.array([-reward])