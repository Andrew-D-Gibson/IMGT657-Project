# Library imports
import random       # Used for random.choice in MCTS
import copy         # Used for deepcopy in MCTS
import time         # Used for optimization
import numpy as np  # Used for stuff and things
import multiprocessing as mp
import tensorflow as tf # Used for neural networks
import pickle

# Class imports
from TicTacToe import TicTacToe
from MCTS import MCTS
from PureNetworkAgent import PureNetworkAgent

# Function imports
from simulate_self_play_games import simulate_self_play_games
from head_to_head_match import head_to_head_match

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

        # Set up the training model
        # This will inherit the weights of the best model before training
        self.training_model = TicTacToe.get_keras_model()

    
    def train_on_examples(self, examples):
        # Pre-allocate numpy arrays for training with keras
        board_input_train = np.empty(((len(examples),) + examples[0][0].shape))
        search_probs_train = np.empty(((len(examples),) + examples[0][1].shape))
        eval_train = np.empty((len(examples), 1))

        # Populate the arrays
        for i, example in enumerate(examples):
            board_input_train[i] = example[0]
            search_probs_train[i] = example[1]
            eval_train[i] = example[2]

        # Define an early stopping callback
        loss_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', 
            patience=config['training']['training_patience'])
        
        # Train the model and record the history                                
        history = self.training_model.fit(
            board_input_train, 
            [search_probs_train, eval_train], 
            callbacks=[loss_callback], 
            epochs=config['training']['training_epochs'], 
            verbose=0)
        
        # Print the training results
        print(" Minibatch Policy Acc: " + 
              str(history.history['policy_output_accuracy'][-1]) + 
              ", Value Acc: " + 
              str(history.history['value_output_accuracy'][-1])) 

    
    def train(self):
        for episode in range(config['training']['num_of_episodes']):
            episode_start_time = time.time()
            
            print("\n\n--------- Episode ", episode, " ---------")

            # Simulate games
            print('Simulating games:')
            new_examples = simulate_self_play_games(self.best_model)
            self.training_examples.extend(new_examples)
            print(f'{len(new_examples)} examples generated, {len(self.training_examples)} total.')

            
            # If we have too many examples, remove the leading examples
            if len(self.training_examples) > config['training']['max_training_examples']:
                self.training_examples = self.training_examples[len(self.training_examples) - config['training']['max_training_examples']:]
            

            # Train the network on minibatches
            print('\nTraining on examples:')   
            self.training_model.set_weights(self.best_model.get_weights())
            for i in range(config['training']['num_of_training_batches']):
                max_minibatch_size = np.min([config['training']['size_of_training_batches'], len(self.training_examples)])
                minibatch = random.sample(self.training_examples, max_minibatch_size)

                print(f"{i+1}/{config['training']['num_of_training_batches']}-{max_minibatch_size}, ", end='')
                self.train_on_examples(minibatch)
            

            # Test network for improvement, and save new network if so
            print('New network against old network (testing for improvement): ')
            wins, draws, losses = head_to_head_match(
                MCTS(self.training_model), 
                MCTS(self.best_model), 
                stochastic = True)
            print(f'W/D/L: {wins} / {draws} / {losses}')

            # Check if the new network is better
            # This doesn't guarantee improvement, but still
            if wins > losses:
                print('Replacing network!  Great success.')
                self.best_model.set_weights(self.training_model.get_weights())
                self.best_model.save(f'Networks/Episode_{episode}')


                print('MCTS with Best network against raw MCTS: ')
                wins, draws, losses = head_to_head_match(
                    MCTS(self.best_model), 
                    MCTS(),
                    stochastic = True)
                print(f'W/D/L: {wins} / {draws} / {losses}')


                print('Raw Best network against raw MCTS: ')
                wins, draws, losses = head_to_head_match(
                    PureNetworkAgent(self.best_model), 
                    MCTS(),
                    stochastic = True)
                print(f'W/D/L: {wins} / {draws} / {losses}')

            else:
                print('Maintaining old network.')


            # Save training examples
            with open(f'TrainingExamples/Episode_{episode}', 'wb') as file:
                pickle.dump(self.training_examples, file)
            
            
            print(f'Total episode time: {time.time() - episode_start_time:.1f} seconds')
            

            
