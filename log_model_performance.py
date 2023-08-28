import os
import csv
from MCTS import MCTS, PureNetworkAgent
from head_to_head_match import head_to_head_match

# Create a new training history file
def initialize_model_log():
    if os.path.exists('Networks/training_history.csv'):
        os.remove('Networks/training_history.csv')

    header = ['Episode_Number', 
                'Network_MCTS_stochastic', 
                'Raw_Network_stochastic',
                'Network_MCTS', 
                'Raw_Network']
    
    with open('Networks/training_history.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        

def log_model_performance(keras_model, episode_num):
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