# train_model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

class StopTrainingCallback(Callback):
    """A custom callback to stop training via an API call."""
    def __init__(self, job_id, jobs_dict):
        super().__init__()
        self.job_id = job_id
        self.jobs_dict = jobs_dict

    def on_epoch_end(self, epoch, logs=None):
        if self.jobs_dict.get(self.job_id, {}).get("status") == "stopping":
            self.model.stop_training = True
            print(f"\nTraining for job {self.job_id} stopped by user.")

# In train_model.py

def create_network(input_data, n_vocab):
    """Creates the structure of the neural network."""
    sequence_length = input_data.shape[1]
    model = Sequential()
    model.add(LSTM(256, input_shape=(sequence_length, 1), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
    return model

def train(model, network_in, network_out, job_id, jobs_dict):
    """Trains the neural network."""
    network_in = np.array(network_in)
    network_out = np.array(network_out)
    network_in = np.reshape(network_in, (network_in.shape[0], network_in.shape[1], 1))
    
    # Save weights with the job_id to keep them unique
    filepath = f"weights-{job_id}.keras" 
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    
    # Create an instance of our new callback
    stop_callback = StopTrainingCallback(job_id, jobs_dict)
    
    # Add our new callback to the list
    callbacks_list = [checkpoint, stop_callback]
    
    # Use fewer epochs for user-uploaded files for faster results
    model.fit(network_in, network_out, epochs=50, batch_size=64, callbacks=callbacks_list)