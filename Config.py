config = {
    'game' : 'TicTacToe',

    'mcts': {
        'exploration_parameter': 4,
        'search_depth': 100,
    },
    
    'self_play': {
        'num_of_episodes': 1000,
        'num_of_self_play_games': 10,
        'num_of_testing_games': 20,
    },
    
    'training': {
        'num_of_training_batches': 32,
        'size_of_training_batches': 1024,
        'training_epochs': 30,
        'training_patience': 4,
        'max_training_examples': 120000,
    }
}