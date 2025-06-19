# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 23:51:18 2024

@author: ammar, samkoesnadi / DDPG-tf2
https://github.com/samkoesnadi/DDPG-tf2/blob/master/src/command_line.py

Run the model in training or testing mode
"""

import logging
import random, time

import gym
from tqdm import trange
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

from ddpg.common_definitions import TOTAL_EPISODES, UNBALANCE_P, MAX_STEP, WARM_UP
from ddpg.model import Brain
from ddpg.utils import Tensorboard
from ddpg.antiroll_env_7_states import AntirollEnv
from src.utils import switch_tab

def train(checkpoints_path=None, load_checkpoints_path=None, warm_up=1, 
          eps_greedy=1.0, use_noise=True, is_train=True):
    """
    We create an environment, create a brain,
    create a Tensorboard, load weights, create metrics,
    create lists to store rewards, and then we run the training loop
    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)


    switch_tab()
    # Step 1. create the gym environment
    env = AntirollEnv(TOTAL_EPISODES, P=0.01, random_rudder=0.6)
    
    num_states = 7
    num_actions = 1
    action_space_high = 0.5
    action_space_low = -0.5

    brain = Brain(num_states, num_actions, 
                  action_space_high, action_space_low)
    
    tf_log_dir = 'D:/Projects/missile-guidance-rl/rl_logging/'
    tensorboard = Tensorboard(log_dir=tf_log_dir)

    # load weights if available
    if load_checkpoints_path is not None:
        logging.info(f"Loading weights from {load_checkpoints_path}, make sure the folder exists")
        brain.load_weights(load_checkpoints_path)

    # all the metrics
    acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
    actions_squared = tf.keras.metrics.Mean('actions', dtype=tf.float32)
    Q_loss = tf.keras.metrics.Mean('Q_loss', dtype=tf.float32)
    A_loss = tf.keras.metrics.Mean('A_loss', dtype=tf.float32)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    # Initialize previous actor loss for comparison
    prev_q_loss = float('inf')

    # run iteration
    with trange(TOTAL_EPISODES) as t:
        for ep in t:
            prev_state = env.reset()
            acc_reward.reset_states()
            actions_squared.reset_states()
            Q_loss.reset_states()
            A_loss.reset_states()
            brain.noise.reset()

            for _ in range(MAX_STEP):

                # Receive state and reward from environment.
                cur_act = brain.act(
                    tf.expand_dims(prev_state, 0),
                    _notrandom=(
                        (ep >= warm_up)
                        and
                        (
                            random.random()
                            <
                            (eps_greedy+(1-eps_greedy)*ep/(TOTAL_EPISODES*1)) # try to mul by 2
                        )
                    ),
                    noise=use_noise
                )
                state, reward, done, _ = env.step(cur_act)
                brain.remember(prev_state, reward, state, int(done))

                # Update weights
                if is_train:
                    c, a = brain.learn(brain.buffer.get_batch(unbalance_p=UNBALANCE_P))
                    Q_loss(c)
                    A_loss(a)

                # Post update for next step
                acc_reward(reward)
                actions_squared(np.square(cur_act/action_space_high))
                prev_state = state

                if done:
                    env.save_data_to_csv()
                    break

            # epsilon greedy linear decay
            eps_greedy = max(0.1, 1.0 - (ep / TOTAL_EPISODES))
            # epsilon greedy exponential decay
            # eps_greedy = 0.1 + (1.0 - 0.1) * np.exp(-0.001 * ep)
            
            ep_reward_list.append(acc_reward.result().numpy())

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
            avg_reward_list.append(avg_reward)

            # Print the average reward
            t.set_postfix(r=avg_reward)
            tensorboard(ep, acc_reward, actions_squared, Q_loss, A_loss)

            # Save weights
            if is_train and ep % 5 == 0:
                current_q_loss = Q_loss.result().numpy()
                if current_q_loss < prev_q_loss:
                    brain.save_weights(f'{checkpoints_path}_best')
                    prev_q_loss = current_q_loss

    env.set_simulation_running_false()
    env.close() # -- there is a problem in the env.close() --> check the thread.join()
    
    if is_train:
        brain.save_weights(checkpoints_path)

    logging.info("Training done...")

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
    
    # print(f'{type(ep_reward_list)}\nep_reward_list: {ep_reward_list}')
    # print(f'{type(avg_reward_list)}\navg_reward_list: {avg_reward_list}')

    # save the reward history    
    values = pd.DataFrame(list(zip(ep_reward_list, avg_reward_list)), 
                          columns=['episode reward', 'average reward'])
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    reward_filename = f"ddpg/new_training/black_box_rl/reward_rl_{timestr}.xlsx"
    with pd.ExcelWriter(reward_filename) as writer:
        values.to_excel(writer, sheet_name="rewards")


if __name__ == "__main__":
    checkpoint_dir = 'D:/Projects/missile-guidance-rl/ddpg/new_training'
    model_name = timestr = time.strftime("%Y%m%d-%H%M%S")
    
    load_checkpoints_path = 'ddpg/new_training/20250517-013256/20250517-013256' # load the latest or None
    
    train(checkpoints_path=f'{checkpoint_dir}/{model_name}',
          load_checkpoints_path=load_checkpoints_path,
          warm_up=1, eps_greedy=1.0,
          use_noise=True, is_train=True)
    

    
    
    
    
    
    
    
    
    