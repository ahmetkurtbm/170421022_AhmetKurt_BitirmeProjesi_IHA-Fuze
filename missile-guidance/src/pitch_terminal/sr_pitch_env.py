# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 20:32:35 2024

@author: ammar

in Simperockets 2 use the 'PhoenixV3 SR2logger for Pitch control' model and flight program

"""

from concurrent.futures import ThreadPoolExecutor
import threading, time, socket, random, math
from tqdm import trange
import pandas as pd
import numpy as np

from src.receiver import readPacket
from pyKey.pyKey_windows import pressKey, releaseKey
from src.utils import relaunch, unpause_game, switch_tab

from antiroll.pid_model import PID_Model
from antiroll.lr_model import LR_Model
from antiroll.lstm_model import LSTM_Model
from antiroll.ddpg_model import DDPG_Model

class SRenv():
    
    def __init__(self, num_episode=1, terminal_pitch_target=-30,
                 max_step =1000, IP="localhost", main_port=2873,
                 show_fps=True):
        
        # class variable
        self.num_ep = num_episode
        self.ep = 0
        self.max_step = max_step
        self.show_fps = show_fps
        self.delta_time = 0.2
        self.saturation = 1
        
        # communication properties            
        self.IP = IP
        self.main_port = main_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(30) # 30 seconds timeout
        self.sock.bind((self.IP, self.main_port))
        
        # global variables
        self.flight_data_stack = [[tuple([0]*14), False]]
        self.simulation_running = True
        self.action = {"Elevator":None,
                       "Aileron": None,
                       "Rudder": None}
        
        # initial position
        self.intial_pos_defined = False
        self.initial_pos_x = None
        self.initial_pos_y = None
        self.initial_pos_z = None
        self.initial_latitude = None
        self.initial_longitude = None

        
        # targets
        self.terminal_pitch_target = terminal_pitch_target
        self.target_latitude = None
        self.target_longitude = None
        
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
        self.antiroll.start()
        time.sleep(2)
        self.observator.start()
        self.pitch_contoller.start()
        self.yaw_controller.start()
        unpause_game()

        # loop_time = time.time()
        with trange(self.num_ep) as t:
            for ep in t:
                self.ep = ep
                self.reset()
                
                grounded = False
                step = 0
                
                while not grounded and step < self.max_step:
                    self.observation_event.wait()
                    data = self.flight_data_stack[-1][0]
        
                    latitude = data[14]
                    longitude = data[15]
                    current_pos = [latitude, longitude]
                    target_pos = [self.target_latitude, self.target_longitude]
                    
                    distance_to_target = self.calculate_distance(current_pos, target_pos)
                    print(f'\ndistance to target: {distance_to_target:.2f}\n')
                    
                    
                    grounded = self.flight_data_stack[-1][1]
                    step += 1
                    
                    
                    # if self.show_fps:
                    #     print('\nFPS {}'.format(1 / (time.time() - loop_time)))
                    #     loop_time = time.time()
                                       
                    
        self.set_simulation_running_false()
        print('main thread ended')
        
    
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
        self.observation_event.set()
    
    def set_missile(self):
        relaunch(end_flight=[125, 170], retry_undo=[520, 627], retry=[517, 569])
    
    def set_simulation_running_false(self):
        self.simulation_running = False
    
    def calculate_target_heading(self, current_pos, target_pos):
        assert len(current_pos) == 2
        assert len(target_pos) == 2
        
        current_lat, current_lon, target_lat, target_lon = map(math.radians, [current_pos[0], current_pos[1], target_pos[0], target_pos[1]])
        delta_lon = target_lon - current_lon
        
        x = math.sin(delta_lon) * math.cos(target_lat)
        y = math.cos(current_lat) * math.sin(target_lat) - math.sin(current_lat) * math.cos(target_lat) * math.cos(delta_lon)
        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        # bearing = (bearing + 360) % 360  # Normalize to [0, 360)
        if bearing > 180:
            bearing -= 360 # Normalize to [-180, 180]
        
        return bearing
    
    def calculate_heading_error(self, current_heading, target_heading):
        heading_error = (target_heading - current_heading + 360) % 360
        if heading_error > 180:
            heading_error -= 360  # Normalize to [-180, 180]
            
        return heading_error
    
    def calculate_distance(self, current_pos, target_pos, radius=1274.2):
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [current_pos[0], current_pos[1], target_pos[0], target_pos[1]])
    
        # Differences in latitude and longitude
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
    
        # Haversine formula
        a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
        # Distance in the specified radius unit
        distance = radius * c
        return distance
        
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
            releaseKey('a')
            releaseKey('d')
        self.rudder_mutex.release()
    
        # print(f'\n---_rudder_control---\nthread name: {thread_name}\nsaturated: {saturated}\n---_rudder_control---\n')

    
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
        
    def rudder_function(self):
        current_time = None
        prev_time = None
        prev_target_heading = None
        with ThreadPoolExecutor(max_workers=self.rudder_worker_num, 
                                thread_name_prefix="Rudder") as rudder:
            while self.simulation_running:
                self.observation_event.wait()
                
                data = self.flight_data_stack[-1][0]
    
                latitude = data[14]
                longitude = data[15]
                yaw = data[10]
                current_pos = [latitude, longitude]
                target_pos = [self.target_latitude, self.target_longitude]
                
                distance_to_target = self.calculate_distance(current_pos, target_pos)
                
                target_heading = self.calculate_target_heading(current_pos, target_pos)
                current_time = time.time()
                
                if prev_time is None or prev_target_heading is None:
                    prev_target_heading = target_heading
                    prev_time = current_time
                    LOS_rate = 0.0
                else:
                    # lambda dot or lambda rate: delta_theta/delta_time
                    delta_heading = target_heading - prev_target_heading
                    delta_time = current_time - prev_time + 0.001
                    LOS_rate = delta_heading / delta_time
                    
                    prev_target_heading = target_heading
                    prev_time = current_time
                    
                # print(f'\ncur_heading: {yaw}\ntarhet_heading: {target_heading}\n')
                
                heading_error = self.calculate_heading_error(yaw, target_heading)
                FPA_rate = -1 * LOS_rate
                rudder_pwm = heading_error * 0.025 + (FPA_rate + 0.000001) * 0.008
                
                rudder.submit(self._rudder_control, rudder_pwm)
                
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
        
        print('antiroll thread ended')
                
    
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
        P = 0.009 # 0.012 # (default value)
        D = 0.01
        with ThreadPoolExecutor(max_workers=self.elevator_worker_num, 
                                thread_name_prefix="Elevator") as elevator:
            while self.simulation_running:
                self.observation_event.wait()
                
                data = self.flight_data_stack[-1][0]
                pitch = data[8]
                pitch_rate = data[9]
                LOS = data[12] # pitch
                
                lt = LOS - self.terminal_pitch_target
                lp = LOS - pitch
                
                pitch_err = lt + lp
                
                elevator_pwm = pitch_err * P - pitch_rate * D
                
                elevator.submit(self._elevator_control, elevator_pwm)
        
        print('elevator thread ended')
    
    def get_initial_postition(self):
        # return tuple([self.initial_pos_x, self.initial_pos_y, self.initial_pos_z])
        return tuple([self.initial_latitude, self.initial_longitude])
    
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
                
                latitude = values.get('latitude')
                longitude = values.get('longitude')
                
                # set the initial position
                if not self.intial_pos_defined:
                    self.initial_pos_x = pos_x
                    self.initial_pos_y = pos_y
                    self.initial_pos_z = pos_z
                    
                    self.initial_latitude = latitude
                    self.initial_longitude = longitude
                    
                    self.intial_pos_defined = True
                    # print(f'initial position:\ninitial_pos_x: {self.initial_pos_x}\ninitial_pos_y: {self.initial_pos_y}\ninitial_pos_z: {self.initial_pos_z}\n')
                    print(f'initial position:\nlatitude: {self.initial_latitude}\nlongitude: {self.initial_longitude}\n')

                altitude = values.get('agl')
                velocity = values.get('velocity')
                ver_vel = values.get('ver_vel')
        
                roll = values.get('roll')
                roll_rate = values.get('roll_rate')
                pitch = values.get('pitch')
                pitch_rate = values.get('pitch_rate')
                yaw = values.get('yaw') if values.get('yaw') <= 180 else (values.get('yaw') - 360)
                yaw_rate = values.get('yaw_rate')
                
                new_pitch = values.get('new_pitch')
                new_yaw = values.get('new_yaw') # this is new heading or the line of sight for the yaw to target.
                
                is_grounded = values.get('grounded') or values.get('destroyed')
                
                self.target_latitude = values.get('target_lat')
                self.target_longitude = values.get('target_long')
                                
                
                self.flight_data_stack.append([tuple([pos_x, pos_y, pos_z, altitude,
                                                      velocity, ver_vel, 
                                                      roll, roll_rate, 
                                                      pitch, pitch_rate, 
                                                      yaw, yaw_rate, 
                                                      new_pitch, new_yaw,
                                                      latitude, longitude]), 
                                                      is_grounded])
                
                text_data = f"""\n----- flight_data -----
target final pitch:  {"{:.3f}".format(self.terminal_pitch_target)}
current LOS:         {"{:.3f}".format(new_pitch)}
LOS - target:        {"{:.3f}".format(new_pitch - self.terminal_pitch_target)}
current pitch:       {"{:.3f}".format(pitch)}
LOS - pitch:         {"{:.3f}".format(new_pitch - pitch)}
error pitch:         {"{:.3f}".format(new_pitch - self.terminal_pitch_target + new_pitch - pitch)}
current heaading     {"{:.3f}".format(new_yaw)}
-----------------------
yaw:                 {"{:.3f}".format(yaw)}
pitch:               {"{:.3f}".format(pitch)}
roll:                {"{:.3f}".format(roll)}

"""
                print(text_data)
                
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
                
        print('observation thread ended...\n')
        self.observation_event.set()
        self.observation_event.clear()
    
    
    def get_flight_data_stack(self):
        return self.flight_data_stack
            
    def save_data_to_csv(self):
        if self.ep == 0:
            self.timestr = time.strftime("%Y%m%d-%H%M%S")
    
        # Extract flight data tuples
        black_box = [data[0] for data in self.flight_data_stack][1:]
        
        # Create DataFrame
        black_box_values = pd.DataFrame(
            black_box,
            columns=[
                'pos_x', 'pos_y', 'pos_z', 'altitude',
                'velocity', 'ver_vel', 'roll', 'roll_rate',
                'pitch', 'pitch_rate', 'yaw', 'yaw_rate',
                'new_pitch', 'new_yaw', 'latitude', 'longitude'
            ]
        )
    
        # Save DataFrame to Excel
        black_box_filename = f"terminal_angle/flight_data/waypoint_black_box_2_{self.timestr}.xlsx"
        mode = 'w' if self.ep == 0 else 'a'
        with pd.ExcelWriter(black_box_filename, mode=mode) as writer:
            black_box_values.to_excel(writer, sheet_name=f"episode_{self.ep}")

                
        # action_history_values = pd.DataFrame(self.action_history, 
        #                                     columns=['aileron_pwm', 'saturated_aileron_pwm'])

        # action_history_filename = f"black_box_rl/action_history_{self.timestr}.xlsx"
        # if self.ep == 1:
        #     with pd.ExcelWriter(action_history_filename) as writer:
        #         action_history_values.to_excel(writer, sheet_name=f"episode_{self.ep}")
        # else:
        #     with pd.ExcelWriter(action_history_filename, mode='a') as writer:
        #         action_history_values.to_excel(writer, sheet_name=f"episode_{self.ep}")

    
    
        
        
        
        
        