# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:31:25 2024

@author: ammar
"""

from controllers.base_model import BaseModel
import joblib

class LR_Model(BaseModel):
    
    def __init__(self, model_name='Linear Regression',
                 model_path='controllers/LinearRegression/models/linear_regression_model.pkl'):
        super().__init__(model_name, model_path)
        
        self.antiroll_linear_reg = joblib.load(model_path)
        
    def get_aileron_pwm(self, state):
        assert len(state) == 2
        
        # self.roll = state[0]
        # self.roll_rate = state[1]
        # state_values = [[self.roll, self.roll_rate]]
        
        self.aileron_pwm = float(self.antiroll_linear_reg.predict([state])[0])
        
        return self.aileron_pwm
    
    def get_model_info(self):
        super().get_model_info()
