# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:53:46 2024

@author: ammar
"""

from antiroll.base_model import BaseModel
from ddpg.model import Brain
import tensorflow as tf

class DDPG_Model(BaseModel):
    
    def __init__(self, model_name='RL DDPG',
                 model_path='ddpg/saved_models/20241102-082455/20241102-082455'):
        super().__init__(model_name, model_path)
        
        num_states = 7
        num_actions = 1
        action_space_high = 0.5
        action_space_low = -0.5

        self.brain = Brain(num_states, num_actions, 
                           action_space_high, action_space_low)
        self.brain.load_weights(self.model_path)
        
    def get_aileron_pwm(self, state):
        assert len(state) == 7
        
        self.aileron_pwm = self.brain.act(tf.expand_dims(state, 0), _notrandom=True,noise=False)[0]
        
        return self.aileron_pwm
    
    def get_model_info(self):
        super().get_model_info()
