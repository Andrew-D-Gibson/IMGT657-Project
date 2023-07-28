import numpy as np
import tensorflow as tf
from GameArchitecture import GameArchitecture

class TicTacToe(GameArchitecture):
    def __init__(self):
        super().__init__()

        # Clear the bitboards
        self.x = 0
        self.o = 0


    # Return a boolean for a particular square being empty
    def check_empty(self, move):
        return (2**move) & (self.x | self.o) == 0
    

    # List the legal moves in the position
    def get_legal_moves(self):
        moves = []
        for i in range(9):
            if self.check_empty(i):
                moves.append(i)
                
        return moves


    # Make a move in the current position
    def make_move(self, move):
        if not self.check_empty(move): 
            raise ValueError('Trying to play in a square that\'s already taken.')
        
        if (self.player_1_move):
            self.x |= (2**move)
        else:
            self.o |= (2**move)

        self.player_1_move = not self.player_1_move


    # Return a boolean, end-value tuple
    # e.g. (False, x) for a game in progress
    # (True, +1) for player 1 win
    # (True, 0) for draw
    # (True, -1) for player 2 win
    def is_game_over(self):
        win_combos = [
            0b111000000, # Bottom row win
            0b000111000, # Middle row win
            0b000000111, # Top row win
            0b100100100, # Right column win
            0b010010010, # Middle column win
            0b001001001, # Left Column Win
            0b100010001, # Top left to bottom right diagonal win
            0b001010100  # Top right to bottom left diagonal win
        ]
        for combo in win_combos:
            if self.x & combo == combo:
                # X won
                return (True, 1)
            elif self.o & combo == combo:
                # O won
                return (True, -1)
        
        # No player has won, so check for any valid moves 
        # e.g. are there any squares still open?
        if 0b111111111 ^ (self.x | self.o) == 0:
            # There are no open squares, so it's a draw
            return (True, 0)
        
        # No player has won and it's not a draw, so the game is not over
        return (False, 0)


    # Debug function for showing the game's state to a user
    def print(self):
        for i in range(2,-1,-1):
            for j in range(2,-1,-1):
                if self.x & 2**((i*3) + j):
                    print(' x ', end = '')
                elif self.o & 2**((i*3) + j):
                    print(' o ', end = '')
                else:
                    print(' _ ', end = '')
                
            if (i == 1):
                """
                if self.is_game_over():
                    if self.value == 1:
                        print('    X Won!', end = '')
                    elif self.value == -1:
                        print('    O Won!', end = '')
                    else:
                        print('    Draw!', end = '')
                el
                """
                if (self.player_1_move):
                    print('    X\'s move', end = '')
                else:
                    print('    O\'s move', end = '')
                    
            print('\n')
        print('\n')


    # Ask for and validate (shouldn't make) a user's move
    def get_user_move(self):
        pass


    # Return the representation of the game's state
    # ready for input to a neural network (defined below)
    def get_keras_input(self):
        array_representation = np.empty((3,3,3))
        
        x_array = [np.uint8(bit) for bit in bin(self.x)[2:]]
        x_array = np.pad(x_array, (9-len(x_array),0))
        array_representation[0,:,:] = np.resize(x_array, (3,3))
        
        o_array = [np.uint8(bit) for bit in bin(self.o)[2:]]
        o_array = np.pad(o_array, (9-len(o_array),0))
        array_representation[1,:,:] = np.resize(o_array, (3,3))
        
        if self.player_1_move:
            array_representation[2,:,:] = np.ones((3,3))
        else:
            array_representation[2,:,:] = np.zeros((3,3))
        
        return array_representation


    # Return a keras learning model for playing this game
    @staticmethod
    def get_keras_model():
        # Define network architecture
        tic_tac_toe_input = tf.keras.layers.Input(shape=(3,3,3), name='TTT_input')
        flatten = tf.keras.layers.Flatten()(tic_tac_toe_input)
        hidden = tf.keras.layers.Dense(64, activation='relu')(flatten)
        hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)
        policy_output = tf.keras.layers.Dense(9, activation='softmax', name='policy_output')(hidden)
        value_output = tf.keras.layers.Dense(1, activation='tanh', name='value_output')(hidden)
        model = tf.keras.models.Model(inputs=tic_tac_toe_input, outputs=[policy_output, value_output])
        
        #model.summary()
        
        # Compile model 
        losses = {
            'policy_output': 'categorical_crossentropy', 
            'value_output': 'mse'
        }
        
        model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
        return model
