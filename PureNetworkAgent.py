import copy
import numpy as np
from TicTacToe import TicTacToe

class PureNetworkAgent():
    def __init__(self, keras_model, parent = None, move = None):
        self.keras_model = keras_model
        self.reset()

        self.parent = parent
        self.move = move

        # If we have a parent, set the board to our parents board
        # and make the move on it to create the new game state
        if self.parent != None:
            self.board = copy.deepcopy(parent.board)
            self.board.make_move(self.move)


    def reset(self):
        self.board = TicTacToe()
        self.children = []


    def search(self):
        self.expand_node()
        board_arrays = np.array([self.board.get_keras_input()])
        
        self.pi, value_est = self.keras_model.predict(board_arrays, verbose=0)

        self.pi = np.squeeze(self.pi)[self.board.get_legal_moves()]  # Get only the search probs for valid moves
        # In regular MCTS, the above would be the search probabilities.
        # Given this is the raw network agent we aren't searching though,
        # so the network's output is used for raw move selection

        # Normalize so we can do stochastic play
        # This doesn't change non-stochastic output
        self.pi /= np.sum(self.pi)


    def expand_node(self):
        self.children = [PureNetworkAgent(keras_model=self.keras_model, parent=self, move=move) 
                                          for move in self.board.get_legal_moves()]
