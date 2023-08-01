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

# Function imports
from simulate_self_play_games import simulate_self_play_games
from head_to_head_match import head_to_head_match

# Config setup
from Config import *


class TrainingManager:
    def __init__(self):
        self.best_model = TicTacToe.get_keras_model()
        self.training_model = TicTacToe.get_keras_model()
        
        self.training_examples = []
 
    
    def train_on_examples(self, examples):
        board_input_train = np.empty(((len(examples),) + examples[0][0].shape))
        search_probs_train = np.empty(((len(examples),) + examples[0][1].shape))
        eval_train = np.empty((len(examples), 1))

        for i, example in enumerate(examples):
            board_input_train[i] = example[0]
            search_probs_train[i] = example[1]
            eval_train[i] = example[2]


        loss_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', 
            patience=config['training']['training_patience'])
                                                    
        history = self.training_model.fit(
            board_input_train, 
            [search_probs_train, eval_train], 
            callbacks=[loss_callback], 
            epochs=config['training']['training_epochs'], 
            verbose=0)
        
        print(" Minibatch Policy Acc: " + 
              str(history.history['policy_output_accuracy'][-1]) + 
              ", Value Acc: " + 
              str(history.history['value_output_accuracy'][-1])) 


    def train(self):
        for episode in range(config['training']['num_of_episodes']):
            episode_start_time = time.time()
            
            print("\n\n--------- Episode ", episode, " ---------")
            

            #tf.keras.models.save_model(self.bestNNet, "BestNNet")   # Make sure we're simulating games with the best network


            # Simulate games
            print('Simulating games...')
            self.training_examples = simulate_self_play_games(self.best_model)
            print(f'{len(self.training_examples)} examples generated.')

            """
            # If we have too many examples, remove the first bunch
            if len(self.trainingExamples) > Config.max_training_examples:
                self.trainingExamples = self.trainingExamples[len(self.trainingExamples) - Config.max_training_examples:]
            """

            # Train the network
            print("\nTraining on examples:")   
            self.train_on_examples(self.training_examples)
            """
            self.training_model.set_weights(self.best_model.get_weights())

            for i in range(config['training']['num_of_training_batches']):
                print(i+1, '/', config['training']['num_of_training_batches'], '-', end='')
                
                max_minibatch_size = np.min([config['training']['size_of_training_batches'], len(self.training_examples)])
                print(max_minibatch_size)
                
                minibatch = np.random.choice(self.training_examples, max_minibatch_size, replace=False)
                
                self.train_on_examples(minibatch)
            """
                
            # Test network for improvement, and save new network if so
            print("New network against old network (testing for improvement): ")
            wins, draws, losses = head_to_head_match(
                MCTS(self.training_model), 
                MCTS(self.best_model), 
                stochastic = False)
            print("W/D/L:", wins, " / ", draws, " / ", losses)

            #if new_wins + (0.5*draws) > (Config.num_of_testing_games * 0.5):
            if wins > losses:
                print("Replacing network!  Great success")
                self.best_model.set_weights(self.training_model.get_weights())
                self.best_model.save("Networks/Episode_" + str(episode))

            else:
                print("Maintaining old network.  Can't compete with the best")


            print("Best network against MCTS: ")
            wins, draws, losses = head_to_head_match(
                MCTS(self.best_model), 
                MCTS(search_depth = config['mcts']['search_depth'] * 2),
                stochastic = False)
            print("W/D/L:", wins, " / ", draws, " / ", losses)
            
            """ 
            # Save training examples
            file = open("TrainingExamples/Episode_" + str(episode), 'wb')
            pickle.dump(self.trainingExamples, file)
            file.close()
            """
            
            print("Total episode time: ", time.time() - episode_start_time)
            

            
