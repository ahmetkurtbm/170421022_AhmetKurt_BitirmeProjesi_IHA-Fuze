# -*- coding: utf-8 -*-
"""
Created on Tue May 20 21:54:55 2025

@author: ammar
"""

# In[] import the models and libs

"""
models:
    - YOLOv8 N
    - YOLOv8 M
"""

# from tqdm import trange
# import matplotlib.pyplot as plt
# import numpy as np
# import time 

from src.yolo_env import Simulation
from src.utils import switch_tab
from pyKey.pyKey_windows import pressKey, releaseKey
import joblib

model_path = "D:\\Projects\\missile-control-with-yolo\\yolo\\YOLOv8_N\\weights\\best.pt"
elevator_model_path = 'D:\\Projects\\missile-control-with-yolo\\elevator_controller\\yolo_linear_regression_model.pkl'

yolo_elevator_model = joblib.load(elevator_model_path)

# sim = Simulation(yolo_elevator_model=yolo_elevator_model, is_nn_model=False)
sim = Simulation()
sim.run()
sim.save_data_to_csv()

# In[] gather data

from src.gather_data_env import Simulation
from src.utils import switch_tab
from pyKey.pyKey_windows import pressKey, releaseKey

model_path = "D:\\Projects\\missile-control-with-yolo\\yolo\\YOLOv8_N\\weights\\best.pt"
sim = Simulation(num_episode=50, yolo_model_path=model_path, 
                 save_video_bool=False, save_train_data=True)
sim.run()

# sim.save_training_data()
