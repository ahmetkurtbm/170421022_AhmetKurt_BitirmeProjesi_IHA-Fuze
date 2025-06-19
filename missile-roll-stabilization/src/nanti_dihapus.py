# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:41:18 2024

@author: ammar

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

from antiroll.benchmarking_env import AntirollEnv
from src.utils import switch_tab
from antiroll.pid_model import PID_Model
from antiroll.lr_model import LR_Model
from antiroll.lstm_model import LSTM_Model
from antiroll.ddpg_model import DDPG_Model
from pyKey.pyKey_windows import pressKey, releaseKey


MAX_STEP = 1000
TOTAL_EPISODES = 17

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

def test(experiments):
    
    switch_tab()
    # Step 1. create the gym environment
    env = AntirollEnv(TOTAL_EPISODES, use_elev=False, use_rudd=False)
    
    # num_states = 2
    # num_actions = 1
    # action_space_high = 0.5
    # action_space_low = -0.5
    
    acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
    
    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []

    # run iteration
    loop_time = time.time()
    with trange(TOTAL_EPISODES) as t:
        for ep in t:
            pressKey('d')
            
             # start activating the antiroll after the 2nd episode
            prev_state = env.reset()
            
            for _ in range(MAX_STEP):
                
                actions = []
                # Receive state and reward from environment.
                if ep >= 2:
                    for model in experiments[ep-2]:
                        actions.append(model.get_aileron_pwm(prev_state))
                else:
                    # the first two episodes do not use any antiroll controller
                    actions.append(0)
                
                cur_act = [np.mean(actions)]               
                state, reward, done, _ = env.step(cur_act)

                # Post update for next step
                acc_reward(reward)
                prev_state = state

                if done:
                    env.save_data_to_csv()
                    break

                print('\nFPS {}'.format(1 / (time.time() - loop_time)))
                print(f'[INFO]:\nepisode: {ep}\nlen model list: {len(experiments[ep-2])}\ncurrent action: {cur_act}')
                loop_time = time.time()
            
            releaseKey('d')
            
            ep_reward_list.append(acc_reward.result().numpy())

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            avg_reward_list.append(avg_reward)

            # Print the average reward
            t.set_postfix(r=avg_reward)
            

    env.set_simulation_running_false()
    env.close() # -- there is a problem in the env.close() --> check the thread.join()

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()


if __name__ == "__main__":
    experiments = []
    
    pid = PID_Model(P=0.014, I=0.001, D=0.012)
    lr = LR_Model()
    lstm = LSTM_Model()
    ddpg = DDPG_Model(model_path='ddpg/saved_models/20241102-082455/20241102-082455')
    
    experiments.append([pid])
    experiments.append([lr])
    experiments.append([lstm])
    experiments.append([ddpg])
    
    experiments.append([pid, lr])
    experiments.append([pid, lstm])
    experiments.append([pid, ddpg])
    experiments.append([lr, lstm])
    experiments.append([lr, ddpg])
    experiments.append([lstm, ddpg])
    experiments.append([pid, lr, lstm])
    experiments.append([pid, lr, ddpg])
    experiments.append([pid, lstm, ddpg])
    experiments.append([lr, lstm, ddpg])
    
    experiments.append([pid, lr, lstm, ddpg]) # ensemble
    
    test(experiments)

##############################################################################
# In[] method list
method_list = [
    "No Antiroll",
    "PID",
    "LR",
    "LSTM",
    "DDPG",
    "PID+LR",
    "PID+LSTM",
    "PID+DDPG",
    "LR+LSTM",
    "LR+DDPG",
    "LSTM+DDPG",
    "PID+LR+LSTM",
    "PID+LR+DDPG",
    "PID+LSTM+DDPG",
    "LR+LSTM+DDPG",
    "Ensemble"
    ]

# In[] understanding the excel produced by after benchmarking.

import pandas as pd

INIT_ALT_THRES = 800
FINAL_ALT_THRES = -10

# excel file names
excel_name_ori = 'data/benchmarking_antiroll/black_box_rl_benchmarking_lat_long.xlsx'
excel_name_keep_alt = 'data/benchmarking_antiroll/black_box_rl_benchmarking_keep_alt.xlsx'

# read the data
doc_ori = pd.read_excel(excel_name_ori, sheet_name=None, usecols=['altitude', 'longitude', 'latitude'])
doc_keep_alt = pd.read_excel(excel_name_keep_alt, sheet_name=None, usecols=['altitude', 'longitude', 'latitude'])

# remove the first sheet becaues it is not used
tmp = list(doc_ori.keys())[0]
doc_ori.pop(tmp)
tmp = list(doc_keep_alt.keys())[0]
doc_keep_alt.pop(tmp)

# some initial and final data are dirty, let's clean them
clear_data = []
for data in doc_ori.values():
    if data['altitude'].iloc[0] < INIT_ALT_THRES:
        data.drop(data.index[0], inplace=True)
    
    if data['altitude'].iloc[-1] < FINAL_ALT_THRES:
        data.drop(data.index[-1], inplace=True)
    
    if data['latitude'].iloc[-1] == 0 and data['altitude'].iloc[-1] == 0:
        data.drop(data.index[-1], inplace=True)
        
    clear_data.append(data)

# normalize longitude and latitude data to start from 0
for data in clear_data:
    init_long = data['longitude'].iloc[0]
    init_lat = data['latitude'].iloc[0]
    
    # Subtract the initial values from all elements in the column
    data['longitude'] = data['longitude'] - init_long
    data['latitude'] = data['latitude'] - init_lat
    
clear_data_keep_alt = []
for data in doc_keep_alt.values():
    if data['altitude'].iloc[0] < INIT_ALT_THRES:
        data.drop(data.index[0], inplace=True)
    
    if data['altitude'].iloc[-1] < FINAL_ALT_THRES:
        data.drop(data.index[-1], inplace=True)
    
    if data['latitude'].iloc[-1] == 0 and data['altitude'].iloc[-1] == 0:
        data.drop(data.index[-1], inplace=True)
        
    clear_data_keep_alt.append(data)

# normalize longitude and latitude data to start from 0
for data in clear_data_keep_alt:
    init_long = data['longitude'].iloc[0]
    init_lat = data['latitude'].iloc[0]
    
    # Subtract the initial values from all elements in the column
    data['longitude'] = data['longitude'] - init_long
    data['latitude'] = data['latitude'] - init_lat
    

# In[] display the graph separately for the original data

import matplotlib.pyplot as plt

# Create subplots
fig, axes = plt.subplots(4, 4, figsize=(15, 15))  # 4x4 grid for 16 plots
axes = axes.flatten()  # Flatten to easily iterate

# Determine global min and max for consistent scaling across plots
min_long = min(data['longitude'].min() for data in clear_data) - 0.01
max_long = max(data['longitude'].max() for data in clear_data) + 0.01
min_lat = min(data['latitude'].min() for data in clear_data) - 0.01
max_lat = max(data['latitude'].max() for data in clear_data) + 0.01

# Plot each DataFrame
for i, data in enumerate(clear_data):
    ax = axes[i]  # Select the correct subplot
    ax.plot(data['longitude'], data['latitude'], linestyle='-')  # Plot with markers
    
    # Set axis limits for consistent scaling
    ax.set_xlim(min_long, max_long)
    ax.set_ylim(min_lat, max_lat)
    
    ax.set_title(method_list[i])  # Set title for each plot
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)  # Add grid for better visualization


# Adjust layout
plt.tight_layout()
plt.show()

# In[] display the original data in one graph
# Create a single figure
plt.figure(figsize=(10, 8))

# # Plot each DataFrame in clear_data
plt.plot(clear_data[0]['longitude'], clear_data[0]['latitude'], linestyle='-', label="No Antiroll")
plt.plot(clear_data[1]['longitude'], clear_data[1]['latitude'], linestyle='-', label="PID")
plt.plot(clear_data[2]['longitude'], clear_data[2]['latitude'], linestyle='-', label="LR")
plt.plot(clear_data[3]['longitude'], clear_data[3]['latitude'], linestyle='-', label="LSTM")
plt.plot(clear_data[4]['longitude'], clear_data[4]['latitude'], linestyle='-', label="DDPG")
plt.plot(clear_data[5]['longitude'], clear_data[5]['latitude'], linestyle='-', label="PID+LR")
plt.plot(clear_data[6]['longitude'], clear_data[6]['latitude'], linestyle='-', label="PID+LSTM")
plt.plot(clear_data[7]['longitude'], clear_data[7]['latitude'], linestyle='-', label="PID+DDPG")
plt.plot(clear_data[8]['longitude'], clear_data[8]['latitude'], linestyle='-', label="LR+LSTM")
plt.plot(clear_data[9]['longitude'], clear_data[9]['latitude'], linestyle='-', label="LR+DDPG")
plt.plot(clear_data[10]['longitude'], clear_data[10]['latitude'], linestyle='-', label="LSTM+DDPG")
plt.plot(clear_data[11]['longitude'], clear_data[11]['latitude'], linestyle='-', label="PID+LR+LSTM")
plt.plot(clear_data[12]['longitude'], clear_data[12]['latitude'], linestyle='-', label="PID+LR+DDPG")
plt.plot(clear_data[13]['longitude'], clear_data[13]['latitude'], linestyle='-', label="PID+LSTM+DDPG")
plt.plot(clear_data[14]['longitude'], clear_data[14]['latitude'], linestyle='-', label="LR+LSTM+DDPG")
plt.plot(clear_data[15]['longitude'], clear_data[15]['latitude'], linestyle='-', label="Ensemble")

# Set labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("All Flight Paths")
plt.legend()  # Show legend for each flight
plt.grid(True)  # Add grid for better visualization
plt.axis("equal")

# Show the plot
plt.show()

# In[] display the graph separately for the data with altitude keeping

import matplotlib.pyplot as plt

# Create subplots
fig, axes = plt.subplots(4, 4, figsize=(15, 15))  # 4x4 grid for 16 plots
axes = axes.flatten()  # Flatten to easily iterate

# Determine global min and max for consistent scaling across plots
min_long = min(data['longitude'].min() for data in clear_data_keep_alt) - 0.01
max_long = max(data['longitude'].max() for data in clear_data_keep_alt) + 0.01
min_lat = min(data['latitude'].min() for data in clear_data_keep_alt) - 0.01
max_lat = max(data['latitude'].max() for data in clear_data_keep_alt) + 0.01

# Plot each DataFrame
for i, data in enumerate(clear_data_keep_alt):
    ax = axes[i]  # Select the correct subplot
    ax.plot(data['longitude'], data['latitude'], linestyle='-')  # Plot with markers
    
    # Set axis limits for consistent scaling
    ax.set_xlim(min_long, max_long)
    ax.set_ylim(min_lat, max_lat)
    
    ax.set_title(method_list[i])  # Set title for each plot
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)  # Add grid for better visualization


# Adjust layout
plt.tight_layout()
plt.show()

# In[] display the data with altitude keeping in one graph
# Create a single figure
plt.figure(figsize=(10, 8))

# # Plot each DataFrame in clear_data
plt.plot(clear_data_keep_alt[0]['longitude'], clear_data_keep_alt[0]['latitude'], linestyle='-', label="No Antiroll")
plt.plot(clear_data_keep_alt[1]['longitude'], clear_data_keep_alt[1]['latitude'], linestyle='-', label="PID")
plt.plot(clear_data_keep_alt[2]['longitude'], clear_data_keep_alt[2]['latitude'], linestyle='-', label="LR")
plt.plot(clear_data_keep_alt[3]['longitude'], clear_data_keep_alt[3]['latitude'], linestyle='-', label="LSTM")
plt.plot(clear_data_keep_alt[4]['longitude'], clear_data_keep_alt[4]['latitude'], linestyle='-', label="DDPG")
plt.plot(clear_data_keep_alt[5]['longitude'], clear_data_keep_alt[5]['latitude'], linestyle='-', label="PID+LR")
plt.plot(clear_data_keep_alt[6]['longitude'], clear_data_keep_alt[6]['latitude'], linestyle='-', label="PID+LSTM")
plt.plot(clear_data_keep_alt[7]['longitude'], clear_data_keep_alt[7]['latitude'], linestyle='-', label="PID+DDPG")
plt.plot(clear_data_keep_alt[8]['longitude'], clear_data_keep_alt[8]['latitude'], linestyle='-', label="LR+LSTM")
plt.plot(clear_data_keep_alt[9]['longitude'], clear_data_keep_alt[9]['latitude'], linestyle='-', label="LR+DDPG")
plt.plot(clear_data_keep_alt[10]['longitude'], clear_data_keep_alt[10]['latitude'], linestyle='-', label="LSTM+DDPG")
plt.plot(clear_data_keep_alt[11]['longitude'], clear_data_keep_alt[11]['latitude'], linestyle='-', label="PID+LR+LSTM")
plt.plot(clear_data_keep_alt[12]['longitude'], clear_data_keep_alt[12]['latitude'], linestyle='-', label="PID+LR+DDPG")
plt.plot(clear_data_keep_alt[13]['longitude'], clear_data_keep_alt[13]['latitude'], linestyle='-', label="PID+LSTM+DDPG")
plt.plot(clear_data_keep_alt[14]['longitude'], clear_data_keep_alt[14]['latitude'], linestyle='-', label="LR+LSTM+DDPG")
plt.plot(clear_data_keep_alt[15]['longitude'], clear_data_keep_alt[15]['latitude'], linestyle='-', label="Ensemble")

# Set labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("All Flight Paths")
plt.legend()  # Show legend for each flight
plt.grid(True)  # Add grid for better visualization
plt.axis("equal")

# Show the plot
plt.show()



# In[] display the data with altitude keeping in one graph (anotherway)
# Create a single figure
plt.figure(figsize=(10, 10))

# Plot each DataFrame in clear_data_keep_alt
for i, data in enumerate(clear_data_keep_alt):
    plt.plot(data['longitude'], data['latitude'], linestyle='-', label=method_list[i])

# Set labels and title
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("All Flight Paths")
plt.legend()  # Show legend for each flight
plt.grid(True)  # Add grid for better visualization

# Set aspect ratio to 1:1
plt.axis("equal")

# Show the plot
plt.show()



# In[] freestyle test for ddpg, etc

import matplotlib.pyplot as plt

from antiroll.pid_model import PID_Model
from antiroll.lr_model import LR_Model
from antiroll.lstm_model import LSTM_Model

USE_DDPG_7_STATES = False

if USE_DDPG_7_STATES:
    from antiroll.benchmarking_env_7_states import AntirollEnv
    from antiroll.ddpg_model_7_states import DDPG_Model
else:
    from antiroll.benchmarking_env import AntirollEnv
    from antiroll.ddpg_model import DDPG_Model

MAX_STEP = 1000
TOTAL_EPISODES = 3

def test(model):
    
    switch_tab()
    # Step 1. create the gym environment
    env = AntirollEnv(use_elev=True, use_rudd=False)
    
    # run iteration
    loop_time = time.time()
    with trange(TOTAL_EPISODES) as t:
        for ep in t:
            
             # start activating the antiroll after the 2nd episode
            prev_state = env.reset()
            
            for _ in range(MAX_STEP):
                                
                cur_act = model.get_aileron_pwm(prev_state)
                print("current act: ", cur_act)
                state, done = env.step([cur_act])

                # Post update for next step
                prev_state = state

                if done:
                    env.save_data_to_csv()
                    break

                print('\nFPS {}'.format(1 / (time.time() - loop_time)))
                # print(f'[INFO]:\nepisode: {ep}\nlen model list: {len(experiments[ep-2])}\ncurrent action: {cur_act}')
                loop_time = time.time()
            
            releaseKey('d')
            
    env.set_simulation_running_false()
    env.close() # -- there is a problem in the env.close() --> check the thread.join()


if __name__ == "__main__":
    experiments = []
    
    # pid = PID_Model(P=0.014, I=0.001, D=0.012)
    # lr = LR_Model()
    # lstm = LSTM_Model()
    ddpg = DDPG_Model(model_path='ddpg/saved_models/20241102-082455/20241102-08245')
    
    # experiments.append([pid])
    # experiments.append([lr])
    # experiments.append([lstm])
    # experiments.append([ddpg])
    
    # experiments.append([pid, lr])
    # experiments.append([pid, lstm])
    # experiments.append([pid, ddpg])
    # experiments.append([lr, lstm])
    # experiments.append([lr, ddpg])
    # experiments.append([lstm, ddpg])
    # experiments.append([pid, lr, lstm])
    # experiments.append([pid, lr, ddpg])
    # experiments.append([pid, lstm, ddpg])
    # experiments.append([lr, lstm, ddpg])
    
    # experiments.append([pid, lr, lstm, ddpg]) # ensemble
    
    test(ddpg)






