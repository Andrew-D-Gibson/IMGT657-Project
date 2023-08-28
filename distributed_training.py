import tensorflow as tf
from ray.air import session, Checkpoint

from TicTacToe import TicTacToe

from Config import *


def distributed_training(model_weights):
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        multi_worker_model = TicTacToe.get_keras_model()
        multi_worker_model.set_weights(model_weights)

    dataset = session.get_dataset_shard('train')

    tf_dataset = dataset.to_tf(
        feature_columns='board_input', 
        label_columns=['policy_output', 'value_output']
    )
    
    # Define an early stopping callback
    loss_callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=config['training']['training_patience'])
        
    multi_worker_model.fit(
        tf_dataset,
        epochs = config['training']['training_epochs'],
        callbacks=[loss_callback],
        verbose = 1
    )
    
    checkpoint = Checkpoint.from_dict(
        dict(model_weights=multi_worker_model.get_weights())
    )
    session.report({}, checkpoint=checkpoint)