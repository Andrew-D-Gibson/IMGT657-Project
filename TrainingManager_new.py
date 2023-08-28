# Library imports
import random       # Used for random.choice in MCTS
import time         # Used for optimization
import numpy as np  # Used for stuff and things
import csv
import tensorflow as tf # Used for neural networks
import pickle
import ray
from ray.air import ScalingConfig
from ray.train.tensorflow import TensorflowTrainer

# Class imports
from TicTacToe import TicTacToe
from MCTS import MCTS

# Function imports
from simulate_self_play_games import simulate_self_play_games
from head_to_head_match import head_to_head_match
from log_model_performance import log_model_performance, initialize_model_log
from distributed_training import distributed_training

# Config setup
from Config import *


class TrainingManager:
    def __init__(self):
        # Load training examples
        if config['training']['old_examples_to_load'] != '':
            with open(f"TrainingExamples/{config['training']['old_examples_to_load']}", 'rb') as file:
                print(f"Loading old training examples from TrainingExamples/{config['training']['old_examples_to_load']}")
                self.training_examples = pickle.load(file)
        else:
            self.training_examples = []

        # Load keras model
        if config['training']['old_network_to_load'] != '':
            print(f"Loading old network from Networks/{config['training']['old_network_to_load']}")
            self.best_model = tf.keras.models.load_model(f"Networks/{config['training']['old_network_to_load']}")
        else:
            self.best_model = TicTacToe.get_keras_model()

        self.best_model.save('Networks/Episode_0')
        self.best_model.save('Networks/Best_Model')

        initialize_model_log()
        #log_model_performance(self.best_model, 0)

    
    def train(self):
        context = ray.init()
        print(context.dashboard_url)

        for episode in range(config['training']['num_of_episodes']):
            # Record the time to monitor how long a particular episode takes
            episode_start_time = time.time()
            

            print("\n\n--------- Episode ", episode, " ---------")


            ## -- Create new training examples by simulating games with self play
            print('Simulating games:')
            example_futures = [simulate_self_play_games.remote() 
                            for _ in range(config['distribution']['cpus_available'])]
            new_examples = ray.get(example_futures)

            # Add the new game lists to the main list
            for example in new_examples:
                self.training_examples.extend(example)
            print(f'{len(self.training_examples)} examples total.')

            
            # If we have too many examples, remove the oldest examples
            if len(self.training_examples) > config['training']['max_training_examples']:
                self.training_examples = self.training_examples[len(self.training_examples) - config['training']['max_training_examples']:]
            
            # Save the new example list
            with open(f'TrainingExamples/Episode_{episode}', 'wb') as file:
                pickle.dump(self.training_examples, file)


            ## -- Perform distributed training
            print('\nTraining on examples:')

            # Create the dataset to distribute
            dataset = ray.data.from_items([{
                'board_input': example[0],
                'policy_output': example[1],
                'value_output': example[2]}
                for example in self.training_examples])

            # Create a TensorFlowTrainer for Ray
            trainer = TensorflowTrainer(
                train_loop_per_worker = distributed_training,
                scaling_config = ScalingConfig(num_workers=config['distribution']['gpus_available'],
                                                use_gpu=True),
                train_loop_config = self.best_model.get_weights(),
                datasets={'train': dataset}
            )

            # Train the model across the attached GPUs
            result = trainer.fit()

            new_model = TicTacToe.get_keras_model()
            new_model.set_weights(result.checkpoint.to_dict()['model_weights'])
            

            ## -- Test new model for improvement
            print('New network against old network: ')
            wins, draws, losses = head_to_head_match(
                MCTS(new_model), 
                MCTS(self.best_model), 
                stochastic = True)
            print(f'\nW/D/L: {wins} / {draws} / {losses}')

            if wins > losses:
                print('Replacing network!  Great success.')
                self.best_model.set_weights(new_model.get_weights())
                self.best_model.save(f'Networks/Episode_{episode}')
                self.best_model.save('Networks/Best_Model')

                log_model_performance(self.best_model, episode)  

            else:
                print('Maintaining old network.')
            
            print(f'Total episode time: {time.time() - episode_start_time:.1f} seconds')
            

            
