# -*- coding: utf-8 -*-
"""
Created on Tue May 20 21:54:55 2025

@author: ammar
"""

# In[] import the models and libs

"""
models:
    - PID (Proportional Integral Derivative)
    - Linear Regression
    - LSTM (long Short-Term Memory)
    - Reinforcement Learning DDPG (Deep Deterministic Policy Gradient)
"""

from tqdm import trange
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time 

from src.benchmarking_env import AntirollEnv
from src.utils import switch_tab
from controllers.PID.pid_model import PID_Model
from controllers.LinearRegression.lr_model import LR_Model
from controllers.LSTM.lstm_model import LSTM_Model
from controllers.DDPG.ddpg_model import DDPG_Model
from pyKey.pyKey_windows import pressKey, releaseKey

"""
1. empty model to prepare the models to be ready
2. yawing to right without antiroll
3. PID (Proportional Integral Derivative)
4. Linear Regression
5. LSTM (long Short-Term Memory)
6. Reinforcement Learning DDPG (Deep Deterministic Policy Gradient)
7. PID + LR
8. PID + LSTM
9. PID + DDPG
10. LR + LSTM
11. LR + DDPG
12. LSTM + DDPG
13. PID + LR + LSTM
14. PID + LR + DDPG
15. PID + LSTM + DDPG
16. LR + LSTM + DDPG
17. Ensemble Models (the combination of PID + LR + LSTM + DDPG)

"""

DDPG_WEIGHT_PATH = 'controllers/DDPG/training_results/20241102-082455/weights/20241102-082455'
models = []

pid = PID_Model(P=0.0014, I=0.00001, D=0.0022)
lr = LR_Model()
lstm = LSTM_Model()
ddpg = DDPG_Model(model_path=DDPG_WEIGHT_PATH)

models.append([pid])
models.append([lr])
models.append([lstm])
models.append([ddpg])

# models.append([pid, lr])
# models.append([pid, lstm])
# models.append([pid, ddpg])
# models.append([lr, lstm])
# models.append([lr, ddpg])
# models.append([lstm, ddpg])
# models.append([pid, lr, lstm])
# models.append([pid, lr, ddpg])
# models.append([pid, lstm, ddpg])
# models.append([lr, lstm, ddpg])

models.append([pid, lr, lstm, ddpg]) # ensemble

# In[] define the method

MAX_STEP = 240 # 4 Hz => 1 minute (60 sec)
TOTAL_EPISODES = len(models) + 2

def evaluate(experiment_models, use_elev=False, use_rudd=False):
    
    switch_tab()
    # Step 1. create the gym environment
    env = AntirollEnv(use_elev=use_elev, use_rudd=use_rudd)
        
    # run iteration
    loop_time = time.time()
    with trange(TOTAL_EPISODES) as t:
        for ep in t:
            # pressKey('d')
            
             # start activating the antiroll after the 2nd episode
            prev_state = env.reset()
            
            for i in range(MAX_STEP):
                
                actions = []
                # Receive state and reward from environment.
                if ep >= 2:
                    for model in experiment_models[ep-2]:
                        actions.append(model.get_aileron_pwm(prev_state))
                else:
                    # the first two episodes do not use any antiroll controller
                    # first episode is garbage, second episode test without controller
                    actions.append(0)
                
                cur_act = [np.mean(actions)]
                state, done = env.step(cur_act)
                prev_state = state

                if done or i >= MAX_STEP - 1:
                    env.save_data_to_csv()
                    break

                print('\nFPS {}'.format(1 / (time.time() - loop_time)))
                print(f'[INFO]:\nepisode: {ep}\nlen model list: {len(experiment_models[ep-2])}\nactions: {actions}\ncurrent action: {cur_act}')
                loop_time = time.time()
            
            # releaseKey('d')

    env.set_simulation_running_false()
    env.close() # -- there is a problem in the env.close() --> check the thread.join()
    

# In[] 1. initial value of 30 degree, calculate:
# 1. Time it takes for roll angle to go from 90% of 30° (≈27°) to 10% of 30° (≈3°).
# 2. Settling Time: Time until system stays within a certain error band (e.g., ±8%)
# 3. Overshoot (%)
# 4. Root Mean Square Error (RMSE): Quantifies deviation from 0 roll
# 5. Max Absolute Error: Worst-case deviation from target

# use Launch Location: benchmarking 01
# initial condition:
# speed: 200 m/s or 0.57 machs
# acceleration: 20 m/s^2
# angle: 30 degree

evaluate(models, use_elev=True)

# In[] 2. targeting a GBAD system from right side (90 degree angle)
# 1. evaluate the roll error average
# 2. target accuracy and missile effectiveness (how far miss from the target)

# use Launch Location: Launch Position 02
# initial condition:
# speed: 215 m/s or 0.63 machs
# acceleration: 65 m/s^2
# angle: 0 degree

import time
from src.yaw90env import SRenv
from controllers.PID.pid_model import PID_Model
from controllers.LinearRegression.lr_model import LR_Model
from controllers.LSTM.lstm_model import LSTM_Model
from controllers.DDPG.ddpg_model import DDPG_Model

from src.utils import relaunch, unpause_game, switch_tab

# PID_Model(P=0.014, I=0.001, D=0.012)

pid_empty = PID_Model(P=0.0014, I=0.00001, D=0.0022)
pid = PID_Model(P=0.014, I=0.001, D=0.012) # this perform better
lr = LR_Model()
lstm = LSTM_Model()
ddpg = DDPG_Model()

models = []
models.append([pid_empty])
# models.append([pid])
# models.append([lr])
# models.append([lstm])
# models.append([ddpg])
models.append([pid, lr, lstm, ddpg])

switch_tab()
time.sleep(0.5)
unpause_game()
time.sleep(1)

for model in models:
    # Step 1. create the gym environment
    env = SRenv(models=model)
    
    # Step 2. Run the simulation
    env.run()
    
    # Step 3. Save the flight data to Excel
    env.save_data_to_csv()
    
    
    relaunch()
    
env.target_latitude
env.target_longitude

# In[] 3. targeting a GBAD system from behind the target(180 degree angle)
# 1. evaluate the roll error average
# 2. target accuracy and missile effectiveness (how far miss from the target)

# use Launch Location: Launch Position 02
# initial condition:
# speed: 215 m/s or 0.63 machs
# acceleration: 65 m/s^2
# angle: 0 degree

import time
from src.yaw180env import SRenv
from controllers.PID.pid_model import PID_Model
from controllers.LinearRegression.lr_model import LR_Model
from controllers.LSTM.lstm_model import LSTM_Model
from controllers.DDPG.ddpg_model import DDPG_Model

from src.utils import relaunch, unpause_game, switch_tab

# PID_Model(P=0.014, I=0.001, D=0.012)

pid_empty = PID_Model(P=0.0014, I=0.00001, D=0.0022)
pid = PID_Model(P=0.014, I=0.001, D=0.012) # this perform better
lr = LR_Model()
lstm = LSTM_Model()
ddpg = DDPG_Model()

models = []
models.append([pid_empty])
# models.append([pid])
# models.append([lr])
# models.append([lstm])
# models.append([ddpg])
models.append([pid, lr, lstm, ddpg])

switch_tab()
time.sleep(0.5)
unpause_game()
time.sleep(1)

for model in models:
    # Step 1. create the gym environment
    env = SRenv(models=model)
    
    # Step 2. Run the simulation
    env.run()
    
    # Step 3. Save the flight data to Excel
    env.save_data_to_csv()
    
    relaunch()


env.target_latitude
env.target_longitude


