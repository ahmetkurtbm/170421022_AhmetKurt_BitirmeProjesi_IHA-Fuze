# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:35:41 2024

@author: ammar
"""

from controllers.base_model import BaseModel
import tensorflow as tf
import numpy as np

class LSTM_Model(BaseModel):
    
    def __init__(self, model_name='LSTM',
                 model_path='controllers/LSTM/models/best_antiroll_model_lstm_64_2_inputs_10t.h5',
                 timestep=10):
        super().__init__(model_name, model_path)
        
        self.antiroll_model = tf.keras.models.load_model(model_path)
        
        # RNN related things
        self.timestep = timestep
        self.num_features = 2
        self.data_for_rnn = np.zeros((self.timestep, 2))
        self.aileron_pwm = 0
        
    def get_aileron_pwm(self, state):
        assert len(state) == 2
        
        state_values = np.array(state)
        
        # Shift the array to make room for the new state (remove the oldest data)
        self.data_for_rnn = np.roll(self.data_for_rnn, -1, axis=0)
        
        # Insert the new state at the last position
        self.data_for_rnn[-1, :] = state_values

        # Reshape the data for RNN
        data_for_rnn = np.reshape(self.data_for_rnn, (1, self.timestep, self.num_features))
        
        # predict the aileron pwm value
        self.aileron_pwm = float(self.antiroll_model.predict(data_for_rnn, verbose=0)[0][0])
        
        return self.aileron_pwm
    
    def get_model_info(self):
        print(f'\n--- MODEL INFO ---\nname: {self.model_name}\ntime step: {self.timestep}')
        self.antiroll_model.summary()
        
    def reset_model(self):
        self.data_for_rnn = np.zeros((self.timestep, 2))
        
