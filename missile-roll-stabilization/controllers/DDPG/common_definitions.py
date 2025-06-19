# -*- coding: utf-8 -*-
"""
@author: samkoesnadi / DDPG-tf2
https://github.com/samkoesnadi/DDPG-tf2/blob/master/src/common_definitions.py

Common definitions of variables that can be used across files
"""

from tensorflow.keras.initializers import glorot_normal  # pylint: disable=no-name-in-module

# brain parameters
GAMMA = 0.99  # for the temporal difference
RHO = 0.001  # to update the target networks
KERNEL_INITIALIZER = glorot_normal()

ACTOR_NODES_1 = 256
ACTOR_NODES_2 = 128
CRITIC_NODES_1 = 256
CRITIC_NODES_2 = 128
CRITIC_NODES_3 = 128
CRITIC_NODES_4 = 128

# buffer params
UNBALANCE_P = 0.8  # newer entries are prioritized
BUFFER_UNBALANCE_GAP = 0.5

# training parameters
STD_DEV = 0.2
BATCH_SIZE = 128
BUFFER_SIZE = 1e6
TOTAL_EPISODES = 100
CRITIC_LR = 1e-3
ACTOR_LR = 25e-4
WARM_UP = 20 # num of warm up epochs
MAX_STEP = 1000

# reward shaping parameters
GLIDING_REWARD = 5
ROLL_PEN = 1
ROLL_MIN = 5
STABLE_ROLL_TIMES = 10
STABLE_REWARD = 10
