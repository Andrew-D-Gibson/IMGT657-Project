import numpy as np
import random
import copy
import tensorflow as tf

from TTTBoard import TTTBoard
from MCTS import MCTS
from trainingExample import trainingExample

from Config import *


def simulateSelfPlayGames():
    trainingExamples = []

    AlphaZeroNetwork = tf.keras.models.load_model("BestNNet")

    for i in range(Config.num_of_self_play_games):

        newTrainingExamples = []
        MCTS = NNetMCTS(NNet = AlphaZeroNetwork)

        while True:
            MCTS.search()

            newTrainingExamples.append(trainingExample(MCTS.board, MCTS.pi))

            childChoice = np.random.choice(len(MCTS.pi), p=MCTS.pi)
            MCTS = MCTS.makeMove(childChoice)

            if MCTS.board.checkGameOver():
                for example in newTrainingExamples:
                    example.addReward(MCTS.board.value)
                    trainingExamples = np.append(trainingExamples, example)
                break     

    return trainingExamples