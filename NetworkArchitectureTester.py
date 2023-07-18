# Library imports
import random       # Used for random.choice in MCTS
import copy         # Used for deepcopy in MCTS
import time         # Used for optimization
import numpy as np  # Used for stuff and things
import multiprocessing as mp
import tensorflow as tf # Used for neural networks
import pickle

from tensorflow.keras.utils import plot_model

# Class imports
from TTTBoard import TTTBoard
from MCTS import MCTS
from trainingExample import trainingExample

# Function imports
from simulateSelfPlayGames import simulateSelfPlayGames

# Config setup
from Config import *


class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv2D(filters, (3,3), strides=strides, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(filters, (3,3), strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization()
        ]

        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, (1,1), strides=strides, padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs 
        for layer in self.main_layers:
            Z = layer(Z)

        skip_Z = inputs 
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)

        # Note that when layers is empty (when stride is 1 and we're not changing dimensionality)
        # skip_Z remains the input. This is where ResNet's magic happens.

        return self.activation(Z + skip_Z)



class NetworkArchitectureTester:
    @staticmethod
    def denseNet():
        ultimate_tic_tac_toe_input = tf.keras.layers.Input(shape=(2,9,9), name='UTTT_input')
        flatten = tf.keras.layers.Flatten()(ultimate_tic_tac_toe_input)

        hidden = tf.keras.layers.Dense(1024, activation='relu')(flatten)
        batchnorm = tf.keras.layers.BatchNormalization()(hidden)

        hidden = tf.keras.layers.Dense(512, activation='relu')(batchnorm)
        batchnorm = tf.keras.layers.BatchNormalization()(hidden)

        hidden = tf.keras.layers.Dense(512, activation='relu')(batchnorm)
        batchnorm = tf.keras.layers.BatchNormalization()(hidden)

        policy_output = tf.keras.layers.Dense(81, activation='softmax', name='policy_output')(batchnorm)
        value_output = tf.keras.layers.Dense(1, activation='tanh', name='value_output')(batchnorm)

        model = tf.keras.models.Model(inputs=ultimate_tic_tac_toe_input, outputs=[policy_output, value_output], name='denseNet')
        
        #model.summary()
        #plot_model(model, to_file='denseNet.png', show_shapes=True, show_layer_names=True)
        
        # Compile model 
        losses = {
            'policy_output': 'categorical_crossentropy', 
            'value_output': 'mse'
        }
        
        model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
        return model


    @staticmethod
    def convNet():
        ultimate_tic_tac_toe_input = tf.keras.layers.Input(shape=(2,9,9), name='UTTT_input')
        reshape = tf.keras.layers.Reshape((9, 9, 2), input_shape=(2,9,9))(ultimate_tic_tac_toe_input)
        
        conv_1 = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', name='conv_1')(reshape)
        batchnorm = tf.keras.layers.BatchNormalization()(conv_1)
        conv_2 = tf.keras.layers.Conv2D(128, (3,3), strides=(3, 3), activation='relu', name='conv_2')(batchnorm)
        batchnorm = tf.keras.layers.BatchNormalization()(conv_2)
        
        flatten = tf.keras.layers.Flatten()(batchnorm)
        
        dense_1 = tf.keras.layers.Dense(512, activation='relu', name='dense_1')(flatten)
        dense_2 = tf.keras.layers.Dense(256, activation='relu', name='dense_2')(dense_1)
        
        policy_output = tf.keras.layers.Dense(81, activation='softmax', name='policy_output')(dense_2)
        value_output = tf.keras.layers.Dense(1, activation='tanh', name='value_output')(dense_2)
        
        model = tf.keras.models.Model(inputs=ultimate_tic_tac_toe_input, outputs=[policy_output, value_output], name='convNet')
        
        #model.summary()
        #plot_model(model, to_file='convNet.png', show_shapes=True, show_layer_names=True)
        
        # Compile model 
        losses = {
            'policy_output': 'categorical_crossentropy', 
            'value_output': 'mse'
        }
        
        model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
        return model


    @staticmethod
    def resNet():
        ultimate_tic_tac_toe_input = tf.keras.layers.Input(shape=(2,9,9), name='UTTT_input')
        reshape = tf.keras.layers.Reshape((9, 9, 2), input_shape=(2,9,9))(ultimate_tic_tac_toe_input)
        conv = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(reshape)
        batchnorm = tf.keras.layers.BatchNormalization()(conv)

        resLayer = batchnorm
        prev_filters = 64
        for filters in [64]*3 + [128]*3:
            strides = 1 if filters == prev_filters else 3
            resLayer = ResidualUnit(filters, strides=strides)(resLayer)
            prev_filters = filters

        flatten = tf.keras.layers.Flatten()(resLayer)
        
        policy_output = tf.keras.layers.Dense(81, activation='softmax', name='policy_output')(flatten)
        value_output = tf.keras.layers.Dense(1, activation='tanh', name='value_output')(flatten)

        model = tf.keras.models.Model(inputs=ultimate_tic_tac_toe_input, outputs=[policy_output, value_output], name='resNet')
        
        #model.summary()
        #plot_model(model, to_file='resNet.png', show_shapes=True, show_layer_names=True)
        
        # Compile model 
        losses = {
            'policy_output': 'categorical_crossentropy', 
            'value_output': 'mse'
        }
        
        model.compile(optimizer='Adam', loss=losses, metrics=['accuracy'])
        return model


    def showModels(self):
        self.denseNet()
        self.convNet()
        self.resNet()


    def loadData(self):
        file = open('MCTS_Examples/MCTSTrainingExamples', 'rb')
        MCTSTrainingExamples = pickle.load(file)
        file.close()

        random.shuffle(MCTSTrainingExamples)
        numOfExamples = len(MCTSTrainingExamples)


        board = np.empty((numOfExamples,2,9,9))
        search_probs = np.empty((numOfExamples,81))
        value = np.empty((numOfExamples,1))

        for i in range(numOfExamples):
            board[i,:,:,:] = MCTSTrainingExamples[i].board_array
            search_probs[i,:] = MCTSTrainingExamples[i].search_probs
            value[i,:] = MCTSTrainingExamples[i].reward


        self.train_board = board[:int(numOfExamples*0.8)]
        self.train_search_probs = search_probs[:int(numOfExamples*0.8)]
        self.train_value = value[:int(numOfExamples*0.8)]

        self.val_board = board[int(numOfExamples*0.8) : int(numOfExamples*0.9)]
        self.val_search_probs = search_probs[int(numOfExamples*0.8) : int(numOfExamples*0.9)]
        self.val_value = value[int(numOfExamples*0.8) : int(numOfExamples*0.9)]

        self.test_board = board[int(numOfExamples*0.9):]
        self.test_search_probs = search_probs[int(numOfExamples*0.9):]
        self.test_value = value[int(numOfExamples*0.9):]


        print("Training Data length: ", len(self.train_value))
        print("Validation Data length: ", len(self.val_value))
        print("Testing Data length: ", len(self.test_value))


    def trainAndTestNetwork(self, network, filename):
        print(filename)
        val_loss_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=Config.training_patience)

        val_data = (self.val_board, [self.val_search_probs, self.val_value])
                                                    
        history = network.fit(self.train_board, [self.train_search_probs, self.train_value], validation_data=val_data, callbacks=[val_loss_callback], epochs=Config.training_epochs, verbose=1)

        tf.keras.models.save_model(network, filename)

        loss, policy_loss, value_loss, policy_acc, value_acc = network.evaluate(self.test_board, [self.test_search_probs, self.test_value], verbose=1)
        print("Evaluated on testing set: ")
        print("Loss: ", loss)
        print("Policy Loss: ", policy_loss)
        print("Value_loss: ", value_loss)
        print("Policy Acc: ", policy_acc)
        print("Value Acc: ", value_acc)
        print("----------")
        print("----------")

        return loss, network


    def testAllNetworks(self):
        denseLoss, denseModel = self.trainAndTestNetwork(self.denseNet(), "DenseNet")
        convLoss, convModel = self.trainAndTestNetwork(self.convNet(), "ConvNet")
        resnetLoss, resModel = self.trainAndTestNetwork(self.resNet(), "ResNet")

        if denseLoss < convLoss and denseLoss < resnetLoss:
            print("Dense Net win")
            self.trainFinalModel(denseModel)
        elif convLoss < resnetLoss:
            print("Conv Net win")
            self.trainFinalModel(convModel)
        else:
            print("Res Net win")
            self.trainFinalModel(resModel)


    def trainFinalModel(self, model):
        file = open('MCTS_Examples/MCTSTrainingExamples', 'rb')
        MCTSTrainingExamples = pickle.load(file)
        file.close()

        random.shuffle(MCTSTrainingExamples)
        numOfExamples = len(MCTSTrainingExamples)


        board = np.empty((numOfExamples,2,9,9))
        search_probs = np.empty((numOfExamples,81))
        value = np.empty((numOfExamples,1))

        for i in range(numOfExamples):
            board[i,:,:,:] = MCTSTrainingExamples[i].board_array
            search_probs[i,:] = MCTSTrainingExamples[i].search_probs
            value[i,:] = MCTSTrainingExamples[i].reward


        loss_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=Config.training_patience)
        model.fit(board, [search_probs, value], callbacks=[loss_callback], epochs=Config.training_epochs, verbose=1)

        tf.keras.models.save_model(model, "BestNNet")