# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:15:00 2024

@author: ammar

base model for antiroll
"""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name, model_path=None):
        self.model_name = model_name
        self.model_path = model_path
        
    @abstractmethod
    def get_aileron_pwm(self, state):
        pass
        
    @abstractmethod
    def get_model_info(self):
        print(f'\n--- MODEL INFO ---\nname: {self.model_name}')
        