# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:14:32 2024

@author: ammar
"""

from gym import spaces
import gym

from collections import deque
import pandas as pd
import numpy as np
import threading, time, socket, random

from src.receiver import readPacket
from pyKey.pyKey_windows import pressKey, releaseKey
from src.utils import relaunch, unpause_game
from ddpg.common_definitions import GLIDING_REWARD, ROLL_PEN, ROLL_MIN, STABLE_REWARD, STABLE_ROLL_TIMES

class AntirollEnv(gym.Env):
    
    def __init__(self, max_episode=1, IP="localhost", main_port=2873, 
                 P=0.01, random_rudder=0.5, use_elev_and_rudd=True):
        
        # RL properties
        self.observation_space = spaces.Box(-np.inf, +np.inf, (2,))
        self.action_space = [0]
        self.num_action_space = len(self.action_space)
        self.prev_shaping = None
        # belong to the action performer thread
        self.action_history = []
        self.delta_time = 0.2   # TODO: observe this
        self.saturation = 1     # TODO: observe this
        self.action_pwm = 0  # should this be None or 0?
        # belong to the observator thread
        self.black_box = [] # belong to the observator thread
        self.state = []
        self.state_history = []
        self.previous_state = None
        self.stable_roll = deque(maxlen=STABLE_ROLL_TIMES)
        
        # communication properties            
        self.IP = IP
        self.main_port = main_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(30) # 30 seconds timeout
        self.sock.bind((self.IP, self.main_port))
        
        # global variables
        self.state_stack = []
        self.simulation_running = True
        self.max_episode = max_episode
        self.episode = 0
        self.history = deque(maxlen=20) # used for checking whether the missile has grounded or not
        self.use_elev_and_rudd = use_elev_and_rudd

        
        # synchronization objects
        self.observation_event = threading.Event()
        self.aileron_mutex = threading.Lock()
        self.aileron_semaphore = threading.Semaphore(value=0)
        self.aileron_threads_communication = {'aileron-01': False,
                                              'aileron-01': False}
        
        # synchronization objects for navigation
        if use_elev_and_rudd:
            self.elevator_mutex = threading.Lock()
            self.elevator_semaphore = threading.Semaphore(value=0)
            self.elevator_threads_communication = {'elevator-01': False,
                                                   'elevator-02': False}
        
        # navigation properties
        self.P = P
        self.random_rudder = random_rudder
        
        # threads initialization
        self.observator = threading.Thread(target=self.get_observation, args=(), name='observator')
        self.aileron_thread_1 = threading.Thread(target=self.aileron, args=(), name='aileron-01')
        self.aileron_thread_2 = threading.Thread(target=self.aileron, args=(), name='aileron-02')
        
        # threads for navigation
        if use_elev_and_rudd:
            self.altitude_evaluator_thread = threading.Thread(target=self.altitude_evaluator, args=(), name='alt-eval')
            self.elevator_thread_1 = threading.Thread(target=self.elevator, args=(), name='elevator-01')
            self.elevator_thread_2 = threading.Thread(target=self.elevator, args=(), name='elevator-02')
            self.rudder_thread = threading.Thread(target=self.rudder, args=(), name='rudder')
        
        # start the threads
        self.observator.start()
        self.aileron_thread_1.start()
        self.aileron_thread_2.start()
        
        if use_elev_and_rudd:
            self.altitude_evaluator_thread.start()
            self.elevator_thread_1.start()
            self.elevator_thread_2.start()
            self.rudder_thread.start()
        
    def reset(self, seed=42):
        super().reset(seed=seed)
        
        self.action_history = []
        self.black_box = []
        self.state_history = []
        self.previous_state = None
        self.action_pwm = 0 # should this be None or 0?
        self.prev_shaping = None
        self.history.clear()
        self.stable_roll.clear()
        
        if self.episode != 0:
            self.set_missile()
        else:
            unpause_game()
        
        self.episode += 1
        # if self.episode <= self.max_episode:
        #     self.simulation_running = True
        # else:
        #     self.simulation_running = False
        
        return self.step(np.array([0]))[0]
    
    def step(self, action_space):
        assert len(self.action_space) == self.num_action_space
        
        # perform the action
        self.action_pwm = action_space[0]
        self.aileron_semaphore.release()
        
        # keep the previous state and retrieve the latest state from state_stack
        self.previous_state = self.state
        
        # wait for the observator thread to retrieve first state data
        self.observation_event.wait()
        state = self.state_stack[-1]
        
        self.state = state[0]
        done = state[1]
        self.state_history.append(self.black_box[-1]) # if there are any data repeated or the length is similar to the black box then remove this
        
        reward = self.calculate_rewards(self.state, self.action_pwm) # state = [roll, roll_rate]
        
        return np.array(self.state), reward, done, {}
        
    def calculate_rewards(self, state, action):
        reward = 0
        
        roll = state[0]
        roll_rate = state[1]
        
        # print('roll:', roll, 'roll rate:', roll_rate)
        # roll dan roll rate kiri (ccw) itu negtive. roll dan roll rate kanan (cw) itu positive
        
        # reward for keeping the missile gliding
        shaping = GLIDING_REWARD
        
        # penalty for error pitch
        shaping = shaping - abs(roll*ROLL_PEN) - abs(roll_rate*ROLL_PEN)
        
        # increasing reward as it keeps the roll angle below abs(ROLL_MIN) = 5
        # shaping = shaping + STABLE_REWARD * (1 / (abs(roll) + 0.0001))
        
        # reward if it can keep the roll between -5 and 5 (C = 5) for the last 20 steps        
        # self.stable_roll.append(abs(roll))
        
        # if len(self.stable_roll) == STABLE_ROLL_TIMES and np.mean(self.stable_roll) <= ROLL_MIN:
        #     print('\n--- succeeded maintaining the roll ---\n')
        #     shaping = shaping + STABLE_REWARD
            
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        return reward
    
    def close(self):
        self.aileron_semaphore.release()
        self.aileron_semaphore.release()
        
        if self.use_elev_and_rudd:
            self.elevator_semaphore.release()
            self.elevator_semaphore.release()
    
    def set_missile(self):
        relaunch(end_flight=[125, 170], retry_undo=[520, 627], retry=[517, 569])
    
    def set_simulation_running_false(self):
        self.simulation_running = False
    
    
##################################################################################
    
    # thread functions
    def aileron(self):
        print(f'{threading.current_thread().name} thread started...\n')
        while self.simulation_running:
            self.aileron_semaphore.acquire()
            self.aileron_mutex.acquire()
            self.aileron_threads_communication[threading.current_thread().name] = True
            self.aileron_mutex.release()
            
            if self.action_pwm > 0:
                sign = 1
                saturated = abs(min(self.delta_time*self.saturation, self.action_pwm))
                pressKey('e')
                time.sleep(saturated)
                # releaseKey('w')
                
            elif self.action_pwm < 0:
                sign = -1
                saturated = abs(max(-(self.delta_time*self.saturation), self.action_pwm))
                pressKey('q')
                time.sleep(saturated)
                # releaseKey('s')
            
            else:
                sign = 1
                saturated = 0
                
                
            self.aileron_mutex.acquire()
            if not self.aileron_threads_communication[self._get_other_aileron_thread_name()]:
                releaseKey('q')
                releaseKey('e')
                
            self.action_history.append(tuple([self.action_pwm, sign * saturated]))
            self.aileron_threads_communication[threading.current_thread().name] = False
            self.aileron_mutex.release()
            
            # print('aileron saturated:', sign * saturated)
            
    
    def get_observation(self):
        print('observation thread started...\n')
        while self.simulation_running:
            try:
                if self.sock and not self.sock._closed:
                    d, a = self.sock.recvfrom(2048) # 2048 maximum bit to receive
                    
                values = readPacket(d)          # extract the data received
                
                pos_x = values.get('pos_x')
                pos_y = values.get('pos_y')
                pos_z = values.get('pos_z')
        
                altitude = values.get('agl')
                velocity = values.get('velocity')
                ver_vel = values.get('ver_vel')
        
                roll = values.get('roll')
                roll_rate = values.get('roll_rate')
                pitch = values.get('pitch')
                pitch_rate = values.get('pitch_rate')
                yaw = values.get('yaw')
                yaw_rate = values.get('yaw_rate')
                
                self.black_box.append(tuple([pos_x, pos_y, pos_z, altitude, velocity, ver_vel, 
                                             roll, roll_rate, pitch, pitch_rate, yaw, yaw_rate]))
                            
                is_grounded = values.get('grounded') or values.get('destroyed')
                
                self.history.append([round(ver_vel, 2), 
                                     round(roll, 2), 
                                     round(pitch, 2)])
                
                if len(self.history) == 20:
                    if self.history[0] == self.history[-1]:
                        is_grounded = True
    
                self.state_stack.append([[roll, roll_rate], is_grounded])
                
                # event set
                self.observation_event.set()
                self.observation_event.clear()
                # print(f"\nobservation gathered: {[[roll, roll_rate], is_grounded]}\n")
                
            except socket.timeout:
                print('\n[ERROR]: TimeoutError at get_observation method\n')
                
                self.black_box.append(tuple([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
                self.history.append([0, 0, 0])
                
                is_grounded = True
                self.state_stack.append([[0, 0], is_grounded])
                
                # event set
                self.observation_event.set()
                self.observation_event.clear()
        
    # TODO: create method for keeping the altitude of the rocket using elevator DONE
    # TODO: add reward to agent if it successfully stabilize the flight
    def altitude_evaluator(self):
        while self.simulation_running:
            self.observation_event.wait()
            
            if len(self.black_box) > 0:
                ver_vel = self.black_box[-1][5]
                # print(f'\n[ALT-EVAL] ver_vel: {ver_vel}\n')
                
                self.elevator_pwm = ver_vel * self.P
                # print(f'\n[ALT-EVAL] elevator_pwm: {self.elevator_pwm}\n')
        
                self.elevator_semaphore.release()
        
        self.elevator_semaphore.release()
        
    def elevator(self):
        while self.simulation_running:
            self.elevator_semaphore.acquire()
            
            self.elevator_mutex.acquire()
            self.elevator_threads_communication[threading.current_thread().name] = True
            self.elevator_mutex.release()
            
            if self.elevator_pwm > 0:
                # sign = 1
                saturated = abs(min(self.delta_time*self.saturation, self.elevator_pwm))
                pressKey('s')
                time.sleep(saturated)
                # releaseKey('s')
                
            elif self.elevator_pwm < 0:
                # sign = -1
                saturated = abs(max(-(self.delta_time*self.saturation), self.elevator_pwm))
                pressKey('w')
                time.sleep(saturated)
                # releaseKey('w')
            
            else:
                # sign = 1
                saturated = 0
                
            self.elevator_mutex.acquire()
            if not self.elevator_threads_communication[self._get_other_elevator_thread_name()]:
                releaseKey('s')
                releaseKey('w')
            # self.elevator_mutex.release()
                
            # self.elevator_mutex.acquire()
            # self.elevator_input_list.append(self.elevator_pwm)
            # self.elevator_input_list_saturated.append(sign * saturated)
            self.elevator_threads_communication[threading.current_thread().name] = False
            self.elevator_mutex.release()
            
    def rudder(self):
        while self.simulation_running:
            
            random_time = random.randint(3, 8)
            if random.random() < self.random_rudder:
                
                if random.randint(0, 1) == 0: # in 0 then left
                    pressKey('a')
                    time.sleep(random_time)
                    releaseKey('a')
                else:
                    pressKey('d')
                    time.sleep(random_time)
                    releaseKey('d')
            else:
                time.sleep(random_time)
            
    
    def _get_other_aileron_thread_name(self):
        if self.aileron_threads_communication[threading.current_thread().name] == 'aileron-01':
            return 'aileron-02'
        else:
            return 'aileron-01'
        
    def _get_other_elevator_thread_name(self):
        if self.elevator_threads_communication[threading.current_thread().name] == 'elevator-01':
            return 'elevator-02'
        else:
            return 'elevator-01'
        
    def save_data_to_csv(self):
        if self.episode == 1:
            self.timestr = time.strftime("%Y%m%d-%H%M%S")


        black_box_values = pd.DataFrame(self.black_box, 
                                        columns=['pos_x', 'pos_y', 'pos_z', 'altitude', 
                                                 'velocity', 'ver_vel', 'roll', 'roll_rate',
                                                 'pitch', 'pitch_rate', 'yaw', 'yaw_rate'])
        black_box_filename = f"black_box_rl/black_box_rl_{self.timestr}.xlsx"
        if self.episode == 1:
            with pd.ExcelWriter(black_box_filename) as writer:
                black_box_values.to_excel(writer, sheet_name=f"episode_{self.episode}")
        else:
            with pd.ExcelWriter(black_box_filename, mode='a') as writer:
                black_box_values.to_excel(writer, sheet_name=f"episode_{self.episode}")
                
                
        state_history_values = pd.DataFrame(self.state_history, 
                                            columns=['pos_x', 'pos_y', 'pos_z', 'altitude', 
                                                     'velocity', 'ver_vel', 'roll', 'roll_rate',
                                                     'pitch', 'pitch_rate', 'yaw', 'yaw_rate'])

        state_history_filename = f"black_box_rl/state_history_{self.timestr}.xlsx"
        if self.episode == 1:
            with pd.ExcelWriter(state_history_filename) as writer:
                state_history_values.to_excel(writer, sheet_name=f"episode_{self.episode}")
        else:
            with pd.ExcelWriter(state_history_filename, mode='a') as writer:
                state_history_values.to_excel(writer, sheet_name=f"episode_{self.episode}")
                
                
        action_history_values = pd.DataFrame(self.action_history, 
                                            columns=['aileron_pwm', 'saturated_aileron_pwm'])

        action_history_filename = f"black_box_rl/action_history_{self.timestr}.xlsx"
        if self.episode == 1:
            with pd.ExcelWriter(action_history_filename) as writer:
                action_history_values.to_excel(writer, sheet_name=f"episode_{self.episode}")
        else:
            with pd.ExcelWriter(action_history_filename, mode='a') as writer:
                action_history_values.to_excel(writer, sheet_name=f"episode_{self.episode}")
                
        
        
        
        
        
        
        
        
        
        
        