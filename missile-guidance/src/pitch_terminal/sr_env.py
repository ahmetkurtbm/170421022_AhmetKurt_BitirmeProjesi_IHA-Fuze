# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:04:26 2024

@author: ammar

in Simperockets 2 use the 'PhoenixV3 SR2logger for Pitch control' model and flight program

"""

# import pandas as pd
import numpy as np
import threading, time, socket, random
from tqdm import trange
from concurrent.futures import ThreadPoolExecutor

from src.receiver import readPacket
from pyKey.pyKey_windows import pressKey, releaseKey
from src.utils import relaunch, unpause_game, switch_tab

from antiroll.pid_model import PID_Model
from antiroll.lr_model import LR_Model
from antiroll.lstm_model import LSTM_Model
from antiroll.ddpg_model import DDPG_Model

class SRenv():
    
    def __init__(self, num_episode=1, max_step =1000, 
                 IP="localhost", main_port=2873,
                 show_fps=True):
        
        # class variable
        self.num_ep = num_episode
        self.ep = 0
        self.max_step = max_step
        self.show_fps = show_fps
        self.delta_time = 0.2   # TODO: observe this
        self.saturation = 1     # TODO: observe this
        
        # communication properties            
        self.IP = IP
        self.main_port = main_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(30) # 30 seconds timeout
        self.sock.bind((self.IP, self.main_port))
        
        # global variables
        self.flight_data_stack = [[0]*14]
        self.simulation_running = True
        self.action = {"Elevator":None,
                       "Aileron": None,
                       "Rudder": None}
        
        # antiroll models
        self.models = []
        self.models.append(PID_Model(P=0.014, I=0.001, D=0.012))
        self.models.append(LR_Model())
        self.models.append(LSTM_Model())
        self.models.append(DDPG_Model(model_path='ddpg/saved_models/20241102-082455/20241102-082455'))
        
        # thread properties
        self.elevator_worker_num = 3
        self.aileron_worker_num = 3
        self.rudder_worker_num = 3

        # synchronization objects
        self.observation_event = threading.Event()
        self.aileron_mutex = threading.Lock()
        self.aileron_threads_communication = {f"Aileron_{i}": threading.Event() for i in range(self.aileron_worker_num)}
        self.elevator_mutex = threading.Lock()
        self.elevator_threads_communication = {f"Elevator_{i}": threading.Event() for i in range(self.elevator_worker_num)}
        self.rudder_mutex = threading.Lock()
        self.rudder_threads_communication = {f"Rudder_{i}": threading.Event() for i in range(self.rudder_worker_num)}
        
        
        
        # threads initialization
        self.antiroll = threading.Thread(target=self.antiroll_function, args=(), name='antiroll')
        self.pitch_contoller = threading.Thread(target=self.elevator_function, args=(), name='pitch_contoller')
        self.yaw_controller = threading.Thread(target=self.rudder_function, args=(), name='yaw_contoller')
        self.observator = threading.Thread(target=self.get_observation, args=(), name='observator')
        
        
    def run(self):
        switch_tab()
        time.sleep(0.5)
        
        # start the threads
        self.observator.start()
        unpause_game()
        self.antiroll.start()
        self.pitch_contoller.start()

        # loop_time = time.time()
        with trange(self.num_ep) as t:
            for ep in t:
                self.ep = ep
                self.reset()
                
                grounded = False
                step = 0
                
                while not grounded and step < self.max_step:
                    self.observation_event.wait()
                    grounded = self.flight_data_stack[-1][1]
                    step += 1
                    
                    
                    
                    # if self.show_fps:
                    #     print('\nFPS {}'.format(1 / (time.time() - loop_time)))
                    #     loop_time = time.time()
                                       
                    
        self.set_simulation_running_false()
        
    
    def reset(self):
        releaseKey('w')
        releaseKey('s')
        releaseKey('a')
        releaseKey('d')
        releaseKey('q')
        releaseKey('e')
        if self.ep != 0:
            self.set_missile()
            self.aileron_threads_communication = [False] * self.aileron_worker_num
    
    def close(self):
        pass
    
    def set_missile(self):
        relaunch(end_flight=[125, 170], retry_undo=[520, 627], retry=[517, 569])
    
    def set_simulation_running_false(self):
        self.simulation_running = False
    
    
# =============================================================================
# thread functions 
# =============================================================================
    
    def _rudder_control(self, pwm):
        thread_name = threading.current_thread().name
    
        self.rudder_mutex.acquire()
        self.rudder_threads_communication[thread_name].set()  # Signal thread communication active
        self.rudder_mutex.release()
    
    
        if pwm > 0:
            sign = 1
            saturated = abs(min(self.delta_time * self.saturation, pwm))
            pressKey('d')
            time.sleep(saturated)
    
        elif pwm < 0:
            sign = -1
            saturated = abs(max(-(self.delta_time * self.saturation), pwm))
            pressKey('a')
            time.sleep(saturated)
    
        else:
            sign = 1
            saturated = 0
    
    
        self.rudder_mutex.acquire()
        self.rudder_threads_communication[thread_name].clear()  # Signal thread communication complete
        if not any(event.is_set() for event in self.rudder_threads_communication.values()):
            releaseKey('w')
            releaseKey('s')
        self.rudder_mutex.release()
    
        # print(f'\n---_rudder_control---\nthread name: {thread_name}\nsaturated: {saturated}')
                
    def rudder_function(self):
        P=0.01
        with ThreadPoolExecutor(max_workers=self.rudder_worker_num, 
                                thread_name_prefix="Rudder") as rudder:
            while self.simulation_running:
                self.observation_event.wait()
                
                data = self.flight_data_stack[-1][0]
                # pitch = data[8]
                # pitch_rate = data[9]
                ver_vel = data[5]
                rudder_pwm = ver_vel * P
                
                rudder.submit(self._rudder_control, rudder_pwm)
    
    def _aileron_control(self, pwm):
        thread_name = threading.current_thread().name
    
        self.aileron_mutex.acquire()
        self.aileron_threads_communication[thread_name].set()  # Signal thread communication active
        self.aileron_mutex.release()
    
    
        if pwm > 0:
            sign = 1
            saturated = abs(min(self.delta_time * self.saturation, pwm))
            pressKey('e')
            time.sleep(saturated)
    
        elif pwm < 0:
            sign = -1
            saturated = abs(max(-(self.delta_time * self.saturation), pwm))
            pressKey('q')
            time.sleep(saturated)
    
        else:
            sign = 1
            saturated = 0
    
    
        self.aileron_mutex.acquire()
        self.aileron_threads_communication[thread_name].clear()  # Signal thread communication complete
        if not any(event.is_set() for event in self.aileron_threads_communication.values()):
            releaseKey('q')
            releaseKey('e')
        self.aileron_mutex.release()
    
        # print(f'\n---_aileron_control---\nthread name: {thread_name}\nsaturated: {saturated}')
                
    def antiroll_function(self):
        with ThreadPoolExecutor(max_workers=self.aileron_worker_num, 
                                thread_name_prefix="Aileron") as aileron:
            while self.simulation_running:
                self.observation_event.wait()
                
                data = self.flight_data_stack[-1][0]
                # roll = data[6]
                # roll_rate = data[7]
                state = data[6:8]
                
                
                actions = []
                for model in self.models:
                    actions.append(model.get_aileron_pwm(state))
                
                aileron_pwm = np.mean(actions)

                # print(f'\n--- antiroll func ---\nroll: {roll}\nroll_rate: {roll_rate}\naileron_pwm: {aileron_pwm}')
                
                aileron.submit(self._aileron_control, aileron_pwm)
                
    
    def _elevator_control(self, pwm):
        thread_name = threading.current_thread().name
    
        self.elevator_mutex.acquire()
        self.elevator_threads_communication[thread_name].set()  # Signal thread communication active
        self.elevator_mutex.release()
    
    
        if pwm > 0:
            sign = 1
            saturated = abs(min(self.delta_time * self.saturation, pwm))
            pressKey('s')
            time.sleep(saturated)
    
        elif pwm < 0:
            sign = -1
            saturated = abs(max(-(self.delta_time * self.saturation), pwm))
            pressKey('w')
            time.sleep(saturated)
    
        else:
            sign = 1
            saturated = 0
    
    
        self.elevator_mutex.acquire()
        self.elevator_threads_communication[thread_name].clear()  # Signal thread communication complete
        if not any(event.is_set() for event in self.elevator_threads_communication.values()):
            releaseKey('w')
            releaseKey('s')
        self.elevator_mutex.release()
    
        # print(f'\n---_elevator_control---\nthread name: {thread_name}\nsaturated: {saturated}')
                
    def elevator_function(self):
        P=0.01
        with ThreadPoolExecutor(max_workers=self.elevator_worker_num, 
                                thread_name_prefix="Elevator") as elevator:
            while self.simulation_running:
                self.observation_event.wait()
                
                data = self.flight_data_stack[-1][0]
                # pitch = data[8]
                # pitch_rate = data[9]
                ver_vel = data[5]
                elevator_pwm = ver_vel * P
                
                elevator.submit(self._elevator_control, elevator_pwm)

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
                
                new_pitch = values.get('new_pitch')
                new_yaw = values.get('new_yaw')
                
                is_grounded = values.get('grounded') or values.get('destroyed')
                                
                
                self.flight_data_stack.append([[pos_x, pos_y, pos_z, altitude,
                                                velocity, ver_vel, 
                                                roll, roll_rate, 
                                                pitch, pitch_rate, 
                                                yaw, yaw_rate, 
                                                new_pitch, new_yaw], 
                                               is_grounded])
                
                # event set
                self.observation_event.set()
                self.observation_event.clear()
                
            except socket.timeout:
                print('\n[ERROR]: TimeoutError at get_observation method\n')
                                
                is_grounded = True
                self.flight_data_stack.append([tuple([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), is_grounded])
                
                # event set
                self.observation_event.set()
                self.observation_event.clear()       