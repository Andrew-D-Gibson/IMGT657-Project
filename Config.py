config = {
    'game' : 'TicTacToe',

    'mcts': {
        'exploration_parameter': 5,
        'search_time': 0.1
    },
    
    'self_play': {
        'num_of_self_play_games': 16,
        'num_of_testing_games': 16,
    },
    
    'training': {
        'num_of_episodes': 10,

        'training_epochs': 8,
        'training_patience': 3,

        'max_training_examples': 7500,

        'old_examples_to_load': '',
        'old_network_to_load': ''
    },

    'distribution': {
        'cpus_available': 32,
        'gpus_available': 1
    }
}