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

# Function imports
from simulate_self_play_games import simulate_self_play_games

# Config setup
from Config import *


class TrainingManager:
    def __init__(self):
        self.best_model = TicTacToe.get_keras_model()
        self.training_model = TicTacToe.get_keras_model()
        
        self.training_examples = []
 
    
    def train_on_examples(self, examples):
        board_train = np.empty((0,2,9,9))
        search_probs_train = np.empty((0,81))
        eval_train = np.empty((0,1))
        
        for example in examples:
            board_train = np.append(board_train, example.board_array[np.newaxis], axis=0)
            search_probs_train = np.append(search_probs_train, example.search_probs[np.newaxis], axis=0)
            eval_train = np.append(eval_train, example.reward[np.newaxis], axis=0)
        
        loss_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=Config.training_patience)
                                                    
        history = self.trainingNNet.fit(board_train, [search_probs_train, eval_train], callbacks=[loss_callback], epochs=Config.training_epochs, verbose=0)
        print(" Minibatch Policy Acc: " + str(history.history['policy_output_accuracy'][-1]) + ", Value Acc: " + str(history.history['value_output_accuracy'][-1])) 

    def log_result(self, result):
        print("Result returned.")
        for r in result:
            self.trainingExamples.append(r)

        
    def train(self):
        for episode in range(Config.num_of_episodes):
            episode_start_time = time.time()
            
            print("\n\n--------- Episode ", episode, " ---------")
            

            tf.keras.models.save_model(self.bestNNet, "BestNNet")   # Make sure we're simulating games with the best network


            # Simulate games
            print("Simulating games: ", end='')

            pool = mp.Pool()
            for i in range(Config.num_of_processes):
                pool.apply_async(simulateSelfPlayGames, args = (), callback = self.log_result)
            pool.close()
            pool.join()

            # If we have too many examples, remove the first bunch
            if len(self.trainingExamples) > Config.max_training_examples:
                self.trainingExamples = self.trainingExamples[len(self.trainingExamples) - Config.max_training_examples:]


            # Train the network
            print("\nTraining on examples:")   
            self.trainingNNet.set_weights(self.bestNNet.get_weights())
            for i in range(Config.num_of_training_batches):
                print(i+1, '/', Config.num_of_training_batches, '-', end='')
                maxMiniBatchSize = np.min([Config.size_of_training_batches, len(self.trainingExamples)])
                exampleMiniBatch = np.random.choice(self.trainingExamples, maxMiniBatchSize, replace=False)
                self.trainOnExamples(exampleMiniBatch)
                   
                
            # Test network for improvement, and save new network if so
            print("New network against old network (testing for improvement): ")
            newWins, draws, oldWins = self.head2headMatch(NNetMCTS(self.trainingNNet), NNetMCTS(self.bestNNet), numOfGames=Config.num_of_testing_games)
            print("W/L/D:", newWins, " / ", oldWins, " / ", draws)

            if newWins + (0.5*draws) > (Config.num_of_testing_games * 0.5):
                print("Replacing network!  Great success")
                self.bestNNet.set_weights(self.trainingNNet.get_weights())
                self.bestNNet.save("Networks/Episode_" + str(episode))

                print("New network against MCTS: ")
                try:
                    wins, draws, losses = self.head2headMatch(NNetMCTS(self.bestNNet), NNetMCTS(rollout=True), numOfGames=Config.num_of_testing_games)
                    print("W/L/D:", wins, " / ", losses, " / ", draws)
                except:
                    print("Error doing self-play games.")

            else:
                print("Maintaining old network.  Can't compete with the best")

                
            # Save training examples
            file = open("TrainingExamples/Episode_" + str(episode), 'wb')
            pickle.dump(self.trainingExamples, file)
            file.close()
            
            print("Total episode time: ", time.time() - episode_start_time)
            

            
