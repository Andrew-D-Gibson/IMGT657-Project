# Base class to define how games are implemented
class GameArchitecture:
    def __init__(self):
        # Set the first player to move when starting a game
        self.player_1_move = True

        # Set the size of the game's policy (same as the neural network's output)
        # This is used for creating training samples later
        self.policy_size = 0
        
    # List the legal moves in the position
    def get_legal_moves(self):
        pass

    # Make a move in the current position
    def make_move(self, move):
        pass

    # Return a boolean, end-value tuple
    # e.g. (False, x) for a game in progress
    # (True, +1) for player 1 win
    # (True, 0) for draw
    # (True, -1) for player 2 win
    def is_game_over(self):
        pass

    # Debug function for showing the game's state to a user
    def print(self):
        pass

    # Ask for and validate (shouldn't make) a user's move
    def get_user_move(self):
        pass

    # Return the representation of the game's state
    # ready for input to a neural network (defined below)
    def get_keras_input(self):
        pass

    # Return a keras learning model for playing this game
    @staticmethod
    def get_keras_model(self):
        pass