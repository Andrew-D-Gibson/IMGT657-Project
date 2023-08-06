config = {
    'game' : 'TicTacToe',

    'mcts': {
        'exploration_parameter': 3,
        'search_time': 1
    },
    
    'self_play': {
        'num_of_self_play_games': 5,
        'num_of_testing_games': 3,
    },
    
    'training': {
        'num_of_episodes': 1,

        'num_of_training_batches': 1,
        'size_of_training_batches': 1024,

        'training_epochs': 500,
        'training_patience': 5,

        'max_training_examples': 120000,
        'old_examples_to_load': 'Episode_0'
    }
}