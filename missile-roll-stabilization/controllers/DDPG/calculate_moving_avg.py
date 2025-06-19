# -*- coding: utf-8 -*-
"""
Created on Fri May 16 21:05:20 2025

@author: ammar
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- Parameters ---
excel_file = 'D:/Projects/missile-guidance-rl/ddpg/saved_models/20241102-082455/reward_rl_20241102-113512.xlsx'
reward_column = 'episode reward'
window_size = 20                  # Change this based on how smooth you want the curve (10, 100)

# --- Load rewards from Excel ---
df = pd.read_excel(excel_file)
rewards = df[reward_column]

# --- Calculate moving average ---
moving_avg = rewards.rolling(window=window_size).mean()

# --- Plotting ---
plt.figure(figsize=(12, 6))
plt.plot(rewards, label='Original Rewards', alpha=0.4)
plt.plot(moving_avg, label=f'{window_size}-Episode Moving Average', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Reward vs Moving Average')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
