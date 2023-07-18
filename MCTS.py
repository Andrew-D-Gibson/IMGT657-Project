import numpy as np
import random
import copy

from TTTBoard import TTTBoard

from Config import *


class MCTS():    
    def __init__(self, NNet = None, parent = None, move = None, search_prob = None):    
        self.reset()
        
        if NNet == None:
            self.rollout_func = self.random_rollout
        else:
            self.rollout_func = self.check_nnet
        
        self.NNet = NNet
        self.parent = parent
        self.move = move
        self.search_prob = search_prob

        
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
    
        # Search probabilities
        self.pi = []
        self.search_probs = []
        
        # Current board position
        self.board = TTTBoard()
        
    
    def random_rollout(self):
        # Check if we're in a terminal position (game is won, lost, or a draw)
        # If it is, backpropagate the value up the tree
        # We 'absolute value' the return to properly punish this node for losing
        # (e.g. A "Game Over" when it's your turn is either a draw or loss, so -1 or 0 reward)
        if self.board.is_game_over():
            self.backpropagate(-np.abs(self.board.value))
            return
        
        potential_children = self.make_children_list()
        
        self.search_probs = np.ones(len(potential_children))    # We treat all children equally in random rollout
        
        random.choice(potential_children).random_rollout()
    
    
    def check_nnet(self):
        # Check if we're in a terminal position (game is won, lost, or a draw)
        # If it is, backpropagate the value up the tree
        # We 'absolute value' the return to properly punish this node for losing
        # (e.g. A "Game Over" when it's your turn is either a draw or loss, so -1 or 0 reward)
        if self.board.is_game_over():
            self.backpropagate(-np.abs(self.board.value))
            return
        
        # Now we know the game isn't over, so return the neural network's evaluation
        board_arrays = np.array([self.board.get_array_representation()])
        
        self.search_probs, value_est = self.NNet.predict(board_arrays, verbose=0)

        self.search_probs = np.squeeze(self.search_probs)[self.board.find_moves()]  # Get only the search probs for valid moves
        value_est = np.squeeze(value_est)
        
        self.backpropagate(value_est)
        
    
    # Top level search method
    def search(self):
        iterations = config['mcts']['search_depth']
                
        for i in range(iterations):
            self.MCTS_iteration()
        
        # Find the number of visits (n) for each node as a proxy for goodness of the node
        # (Not the highest average value, to make sure we're confident)
        children_n = [child.n for child in self.children]
        
        self.pi = [child_n / sum(children_n) for child_n in children_n]
        
    
    def make_move(self, move = None):
        if move not in self.board.find_moves():
            print('No valid move given!')
            return
        
        if len(self.children) == 0:
            self.board.make_move(move)
            return self
            
        for child in self.children:
            if child.move == move:
                child.parent = None
                return child
            
        print('Error! make_move didn\'t return a new node.')  # In theory should never see this. In theory.
        
        
    # Returns a list of all the valid children nodes
    def make_children_list(self):
        return [MCTS(NNet=self.NNet, parent=self, move=move) for move in self.board.find_moves()]
        
    
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
        if self.board.is_game_over():
            self.backpropagate(-np.abs(self.board.value))    # Prior node won or drew, so reward it with +1 (or 0 so w/e)
            return
            
        # Check if we're at a leaf node (no children exist in this tree)
        if self.children == []:     
            # If we've never been here before (n = 0) then perform rollout
            if self.n == 0:
                self.rollout_func()
            
            # Otherwise extend the tree by finding all the children and then 
            # performing rollout on the first child
            else:
                self.children = self.make_children_list()
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
            