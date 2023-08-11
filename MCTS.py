import numpy as np
import random
import copy
import time

from TicTacToe import TicTacToe

from Config import *


class MCTS():    
    def __init__(self, keras_model = None, parent = None, move = None):  
        self.reset()

        if keras_model == None:
            self.rollout_func = self.random_rollout
        else:
            self.rollout_func = self.check_keras_model
        self.keras_model = keras_model   

        self.parent = parent
        self.move = move

        
        # If we have a parent, set the board to our parents board
        # and make the move on it to create the new game state
        if self.parent != None:
            self.board = copy.deepcopy(parent.board)
            self.board.make_move(self.move)

            
    def reset(self):
        # Set the node's total value to 0
        self.t = 0
        
        # Set the node's total visits to 0
        self.n = 0
 
        # Children nodes (also of class NNetMCTS)
        self.children = []

        # Remove anything upstream
        self.parent = None
        self.move = None
    
        # Search probabilities
        self.pi = []
        self.search_probs = []
        
        # Current board position
        if config['game'] == 'TicTacToe':
            self.board = TicTacToe()
        else:
            raise ValueError('Config file missing a proper game to play.')
        
    
    def random_rollout(self):
        # Check if we're in a terminal position (game is won, lost, or a draw)
        # If it is, backpropagate the value up the tree
        # We 'absolute value' the return to properly punish this node for losing
        # (e.g. A "Game Over" when it's your turn is either a draw or loss, so -1 or 0 reward)
        game_over, value = self.board.is_game_over()
        if game_over:
            self.backpropagate(-np.abs(value))
            return
        
        self.expand_node()
        
        self.search_probs = np.ones(len(self.children))    # We treat all children equally in random rollout
        
        random.choice(self.children).random_rollout()
    
    
    def check_keras_model(self):
        # Check if we're in a terminal position (game is won, lost, or a draw)
        # If it is, backpropagate the value up the tree
        # We 'absolute value' the return to properly punish this node for losing
        # (e.g. A "Game Over" when it's your turn is either a draw or loss, so -1 or 0 reward)
        game_over, value = self.board.is_game_over()
        if game_over:
            self.backpropagate(-np.abs(value))
            return
        
        # Now we know the game isn't over, so return the neural network's evaluation
        board_arrays = np.array([self.board.get_keras_input()])
        
        self.search_probs, value_est = self.keras_model.predict(board_arrays, verbose=0)

        self.search_probs = np.squeeze(self.search_probs)[self.board.get_legal_moves()]  # Get only the search probs for valid moves
        value_est = np.squeeze(value_est)
        
        self.backpropagate(value_est)
        
    
    # Top level search method
    def search(self, depth_to_search = -1, seconds_to_search = -1):
        # Depth based search (should be used rarely)
        if depth_to_search != -1:
            for i in range(depth_to_search):
                self.MCTS_iteration()

        # Time based search
        else:
            if seconds_to_search == -1:
                seconds_to_search = config['mcts']['search_time']

            # Tensorflow has a weird startup timing,
            # so we get it going before starting the clock
            self.MCTS_iteration()

            start_time = time.time()
            while time.time() - start_time < seconds_to_search:
                self.MCTS_iteration()
            
        
        # Find the number of visits (n) for each node as a proxy for goodness of the node
        # (Not the highest average value, to make sure we're confident)
        children_n = [child.n for child in self.children]
        
        self.pi = [child_n / sum(children_n) for child_n in children_n]
        
        
    # Creates a list of all the valid children nodes under this node
    def expand_node(self):
        self.children = [self.__class__(keras_model=self.keras_model, parent=self, move=move) 
                              for move in self.board.get_legal_moves()]
        
    
    # Backpropagates a states value back up through all the parent nodes,
    # flipping it's sign every time b/c of the tree's adversarial nature.
    # (e.g. a good node for O is a bad node for X)
    def backpropagate(self, value):
        self.t += value    
        self.n += 1
        
        if self.parent != None:
            self.parent.backpropagate(-value)
            
    
    # Performs a single iteration of a Monte-Carlo tree search
    def MCTS_iteration(self):
        # Check if we're at a terminal node (no more game to play because it's over!)
        # If so, then just backpropagate the value of the position
        game_over, value = self.board.is_game_over()
        if game_over:
            self.backpropagate(-np.abs(value))    # Prior node won or drew, so reward it with +1 (or 0 so w/e)
            
        # Check if we're at a leaf node (no children exist in this tree)
        elif len(self.children) == 0:     
            # If we've never been here before (n = 0) then perform rollout
            if self.n == 0:
                self.rollout_func()
            
            # Otherwise extend the tree by finding all the children and then 
            # performing rollout on the first child
            else:
                self.expand_node()
                self.children[0].rollout_func()
        
        # The game isn't over and we're not at a leaf node, therefore we're somewhere in the search tree.
        # Pick the child that maximises UCB and continue search
        else:
            children_UCB = []
            
            for child, search_prob in zip(self.children, self.search_probs):
                value_term = child.t / (child.n or 1) # Average value (total value / number of times visited)
                exploration_term = search_prob * (np.sqrt(self.n) / (1 + child.n))
                children_UCB.append(-value_term + (config['mcts']['exploration_parameter'] * exploration_term))
                # The -value_term is because this node is its child node's opponent
                
            best_UCB = np.argmax(children_UCB)
            
            self.children[best_UCB].MCTS_iteration()


    # Debug method for double checking the MCTS functionality
    def print(self):
        print('MCTS Node:')

        self.board.print()

        print(f'Total Value: {self.t}')
        print(f'Node Visits: {self.n}')


        def extend_list(sub_list):
            legal_moves = self.board.get_legal_moves()
            extended_list = ['-'] * self.board.policy_size
 
            while len(sub_list) > 0:
                extended_list[legal_moves[0]] = sub_list[0]
                
                legal_moves.pop(0)
                sub_list.pop(0)

            return extended_list
            
        def print_extended_list(list_to_print):
            extended_list = extend_list(list_to_print)
            for i in range(2, -1, -1):
                for j in range(2, -1, -1):
                    if extended_list[(i*3) + j] == '-':
                        print('-\t', end='')
                        continue

                    print(f'{extended_list[(i*3) + j]:.2f}', end='\t')
                print('\n')

        print('\nRaw Search Probs:')
        print_extended_list(list(self.search_probs))

        print('\nPi:')
        print_extended_list(self.pi)


class PureNetworkAgent(MCTS):
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


# BROKEN MESS OF SHIT
class PlayerAgent(MCTS):
    def search(self):
        self.expand_node()
        self.board.print()

        while True:
            user_move = input('Move to play? (0-8): ')
            user_move = int(user_move)

            if user_move in self.board.get_legal_moves():
                break
            
            print(f'Not a legal move! Legal moves are: {self.board.get_legal_moves()}')

        self.pi = np.zeros(self.board.policy_size)
        self.pi[self.board.get_legal_moves().index(user_move)] = 1

