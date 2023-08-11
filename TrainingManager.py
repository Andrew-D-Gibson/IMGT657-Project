# Library imports
import random       # Used for random.choice in MCTS
import time         # Used for optimization
import numpy as np  # Used for stuff and things
import csv
import tensorflow as tf # Used for neural networks
import pickle
import ray

# Class imports
from TicTacToe import TicTacToe
from MCTS import MCTS, PureNetworkAgent

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

        self.best_model.save('Networks/Episode_0')
        self.best_model.save('Networks/Best_Model')

        # Set up the training model
        # This will inherit the weights of the best model before training
        self.training_model = TicTacToe.get_keras_model()

        # Create a new training history file
        header = ['Episode_Number', 
                  'Network_MCTS_stochastic', 
                  'Raw_Network_stochastic',
                  'Network_MCTS', 
                  'Raw_Network']
        
        with open('Networks/training_history.csv', mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header)  # Write the header
        
        #self.add_model_to_history(self.best_model, 0)


    def add_model_to_history(self, keras_model, episode_num):
        print('Adding model to training history:')
        new_history_record = [episode_num]

        print('MCTS with network against raw MCTS: ')
        wins, draws, losses = head_to_head_match(
            MCTS(keras_model), 
            MCTS(),
            stochastic = True)
        print(f'\nW/D/L: {wins} / {draws} / {losses}')
        new_history_record.extend([(wins, draws, losses)])

        print('Raw network against raw MCTS: ')
        wins, draws, losses = head_to_head_match(
            PureNetworkAgent(keras_model), 
            MCTS(),
            stochastic = True)
        print(f'\nW/D/L: {wins} / {draws} / {losses}')
        new_history_record.extend([(wins, draws, losses)])

        print('Non-Stochastic MCTS with network against raw MCTS: ')
        wins, draws, losses = head_to_head_match(
            MCTS(keras_model), 
            MCTS(),
            stochastic = False,
            num_of_games = 2)
        print(f'\nW/D/L: {wins} / {draws} / {losses}')
        new_history_record.extend([(wins, draws, losses)])

        print('Non-Stochastic Raw network against raw MCTS: ')
        wins, draws, losses = head_to_head_match(
            PureNetworkAgent(keras_model), 
            MCTS(),
            stochastic = False,
            num_of_games = 2)
        print(f'\nW/D/L: {wins} / {draws} / {losses}')
        new_history_record.extend([(wins, draws, losses)])

        with open('Networks/training_history.csv', mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(new_history_record)

    
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
        context = ray.init()
        print(context.dashboard_url)

        for episode in range(config['training']['num_of_episodes']):
            episode_start_time = time.time()
            
            print("\n\n--------- Episode ", episode, " ---------")


            # Simulate games
            print('Simulating games:')
            new_examples = [simulate_self_play_games.remote() for _ in range(16)]
            new_examples = ray.get(new_examples)

            for example in new_examples:
                self.training_examples.extend(example)
            print(f'{len(self.training_examples)} examples total.')

            
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
            print(f'\nW/D/L: {wins} / {draws} / {losses}')

            # Check if the new network is better
            # This doesn't guarantee improvement, but still
            if wins > losses:
                print('Replacing network!  Great success.')
                self.best_model.set_weights(self.training_model.get_weights())
                self.best_model.save(f'Networks/Episode_{episode}')
                self.best_model.save('Networks/Best_Model')

                self.add_model_to_history(self.best_model, episode)  

            else:
                print('Maintaining old network.')


            # Save training examples
            with open(f'TrainingExamples/Episode_{episode}', 'wb') as file:
                pickle.dump(self.training_examples, file)
            
            
            print(f'Total episode time: {time.time() - episode_start_time:.1f} seconds')
            

            
