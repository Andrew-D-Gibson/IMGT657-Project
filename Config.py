config = {
    'game' : 'TicTacToe',

    'mcts': {
        'exploration_parameter': 4,
        'search_depth': 10,
    },
    
    'self_play': {
        'num_of_self_play_games': 10,
        'num_of_testing_games': 20,
    },
    
    'training': {
        'num_of_episodes': 2,
        'num_of_training_batches': 2,
        'size_of_training_batches': 1024,
        'training_epochs': 50,
        'training_patience': 5,
        'max_training_examples': 120000,
    }
}