config = {
    'game' : 'TicTacToe',

    'mcts': {
        'exploration_parameter': 5,
        'search_time': 0.2
    },
    
    'self_play': {
        'num_of_self_play_games': 8,
        'num_of_testing_games': 16,
    },
    
    'training': {
        'num_of_episodes': 10,

        'num_of_training_batches': 8,
        'size_of_training_batches': 1024,

        'training_epochs': 500,
        'training_patience': 5,

        'max_training_examples': 120000,

        'old_examples_to_load': '',
        'old_network_to_load': ''
    }
}