# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:14:31 2024

@author: ammar
"""

from controllers.base_model import BaseModel

class PID_Model(BaseModel):
    
    def __init__(self, P, I, D, model_name='PID'):
        super().__init__(model_name)
        
        self.P = P
        self.I = I
        self.D = D
        
        self.roll = 0
        self.roll_rate = 0
        self.accumulated_roll_error = 0
        
    def get_aileron_pwm(self, state):
        assert len(state) == 2
        
        self.roll = state[0]
        self.roll_rate = state[1]
        
        
        self.roll_error = -self.roll
        self.accumulated_roll_error += self.roll_error
        self.aileron_pwm = (self.roll_error * self.P + 
                            self.accumulated_roll_error * self.I + 
                            (self.roll_rate + 0.000001) * self.D) * 0.5
        
        return self.aileron_pwm
    
    def get_model_info(self):
        print(f'\n--- MODEL INFO ---\nname: {self.model_name}\nP: {self.P}\tI: {self.I}\tD: {self.D}')
