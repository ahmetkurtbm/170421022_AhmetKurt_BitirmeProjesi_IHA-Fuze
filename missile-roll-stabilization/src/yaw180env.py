# -*- coding: utf-8 -*-
"""
Created on Thu May 22 20:13:35 2025


@author: ammar

in Simperockets 2 use the 'PhoenixV3 SR2logger for Pitch control' model and flight program

The missile's initial position is assumed to be pointing in the direction of the target

currently working on "Launch Position 02"
"""

from concurrent.futures import ThreadPoolExecutor
import threading, time, socket, math
from tqdm import trange
import pandas as pd
import numpy as np

from src.receiver import readPacket
from pyKey.pyKey_windows import pressKey, releaseKey
from src.utils import relaunch

from controllers.PID.pid_model import PID_Model

DESIRED_STRAIGHT_DIST = 20 # meter (we need to convert this to degree)
RADIUS_PLANET = 1274200 # meter
STRAIGHT_DIST = math.degrees(DESIRED_STRAIGHT_DIST / RADIUS_PLANET)

DIAMETER_TURNING_POINT = 0.08 # in latitude diff

class SRenv():
    
    def __init__(self, num_episode=1, max_step =1000, 
                 IP="localhost", main_port=2873, 
                 models=[PID_Model(P=0.014, I=0.001, D=0.012)],
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
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # allow re-binding
        self.sock.settimeout(30) # 30 seconds timeout
        self.sock.bind((self.IP, self.main_port))
        
        # global variables
        self.flight_data_stack = [[tuple([0]*14), False]]
        self.simulation_running = True
        self.action = {"Elevator":None,
                       "Aileron": None,
                       "Rudder": None}
        self.action_history = {"pitch": [],
                               "roll": [],
                               "yaw": []}
        
        # initial position
        self.intial_pos_defined = False
        self.initial_pos_x = None
        self.initial_pos_y = None
        self.initial_pos_z = None
        self.initial_latitude = None
        self.initial_longitude = None
        
        # target and waypoint position (latitude, longitude)
        self.target_latitude = None
        self.target_longitude = None
        self.straight_dist_lat = None
        self.straight_dist_long = None
        self.turning_point_lat = None
        self.turning_point_long = None
        self.waypoints = [(self.turning_point_lat, self.turning_point_long),
                          (self.straight_dist_lat, self.straight_dist_long),
                          (self.target_latitude, self.target_longitude)]
        self.waypoint_index = 0
        self.distance_to_waypoint = math.inf

        
        # antiroll models
        self.models = models
        self.model_name = models[0].model_name if len(models) == 1 else "_".join([m.model_name for m in models])
        
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
        # self.yaw_controller = threading.Thread(target=self.rudder_function, args=(), name='yaw_contoller')
        self.observator = threading.Thread(target=self.get_observation, args=(), name='observator')
        self.mission_autopilot_thread = threading.Thread(target=self.mission_autopilot, args=(), name='autopilot')
        
    def run(self):
        # start the threads
        self.observator.start()
        self.antiroll.start()
        self.pitch_contoller.start()
        self.mission_autopilot_thread.start()

        # loop_time = time.time()
        with trange(self.num_ep) as t:
            for ep in t:
                self.ep = ep
                self.reset()
                
                grounded = False
                step = 0
                
                while not grounded and step < self.max_step and self.waypoint_index < len(self.waypoints):
                    self.observation_event.wait()
                    grounded = self.flight_data_stack[-1][1]
                    step += 1
                    
                    # if self.show_fps:
                    #     print('\nFPS {}'.format(1 / (time.time() - loop_time)))
                    #     loop_time = time.time()
                                       
                    
        self.set_simulation_running_false()
        print('main thread ended')
        # self.sock.close()
        # print('socket closed')
        
    
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
    
    def mission_autopilot(self):
        Pr = 0.032
        Dr = 0.012
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
                
                # print(f'Position: ({latitude:.2f}, {longitude}:.2f)')
                # print(f'Normalized: ({(latitude-self.initial_latitude)*10000:.2f}, {(longitude-self.initial_longitude)*10000:.2f})')
                
                if self.waypoints[0][0] is not None:
                    # print("self.waypoint: ", self.waypoints)
                    self.distance_to_waypoint = self.calculate_distance(current_pos, self.waypoints[self.waypoint_index])
                    if self.distance_to_waypoint <= 0.2 and self.waypoint_index < len(self.waypoints)-1:
                        print(f'\n############\nWAYPOINT CHANGED - distance: {self.distance_to_waypoint:.2f}\n############\n')
                        self.waypoint_index += 1
                        # if self.waypoint_index > len(self.waypoints)-1:
                        #     self.simulation_running = False
                        #     break
                    

                    target_heading = self.calculate_target_heading(current_pos, self.waypoints[self.waypoint_index])
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
                    
                    heading_error = self.calculate_heading_error(yaw, target_heading)
                    FPA_rate = -1 * LOS_rate
                    rudder_pwm = heading_error * Pr + (FPA_rate + 0.000001) * Dr
                    
                    
                    print(f'---\nWAYPOINT - {self.waypoint_index}\ncurrent heading: {yaw:.2f}\nheading_error: {heading_error:.2f}\ndistance_to_waypoint: {self.distance_to_waypoint:.2f}\nnext waypoint (lat, long): ({self.waypoints[self.waypoint_index][0]}, {self.waypoints[self.waypoint_index][1]})\ncurrent_pos: {current_pos}\n')
                    rudder.submit(self._rudder_control, rudder_pwm)
        print('mission autopilot thread ended')
                    
    
    def calculate_target_heading(self, current_pos, target_pos):
        assert len(current_pos) == 2
        assert len(target_pos) == 2
        
        current_lat, current_lon, target_lat, target_lon = map(math.radians, [current_pos[0], current_pos[1], target_pos[0], target_pos[1]])
        delta_lon = target_lon - current_lon
        
        x = math.sin(delta_lon) * math.cos(target_lat)
        y = ( math.cos(current_lat) * math.sin(target_lat) ) - ( math.sin(current_lat) * math.cos(target_lat) * math.cos(delta_lon) )
        bearing = math.atan2(x, y)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360  # Normalize to [0, 360)
        
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
        a = math.sin(delta_lat / 2)**2 + ( math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2)**2 )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
        # Distance in the specified radius unit
        distance = radius * c
        return distance
    
    def get_direction(self, heading):
        true_heading = (heading + 360) % 360
        
        if (315 <= true_heading < 360) or (0 <= true_heading <= 45):
            return 'N'
        elif 45 < true_heading <= 135:
            return 'E'
        elif 135 < true_heading <= 225:
            return 'S'
        else:
            return 'W'

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
        self.action_history["yaw"].append(tuple([pwm, sign * saturated]))
        self.rudder_mutex.release()
        
        # print(f'\n---_rudder_control---\nthread name: {thread_name}\nsaturated: {saturated}\n---_rudder_control---\n')
                
    def rudder_function(self):
        # currently the rudder function is not used because we use mission autopilot
        P=0.01
        with ThreadPoolExecutor(max_workers=self.rudder_worker_num, 
                                thread_name_prefix="Rudder") as rudder:
            while self.simulation_running:
                self.observation_event.wait()
                
                rudder_pwm =  0 * P
                
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
        self.action_history["roll"].append(tuple([pwm, sign * saturated]))
        self.aileron_mutex.release()

        # print(f'\n---_aileron_control---\nthread name: {thread_name}\nsaturated: {saturated}')
                
    def antiroll_function(self):
        with ThreadPoolExecutor(max_workers=self.aileron_worker_num, 
                                thread_name_prefix="Aileron") as aileron:
            while self.simulation_running:
                self.observation_event.wait()
                
                data = self.flight_data_stack[-1][0]
                state = data[6:8]
                
                
                actions = []
                for model in self.models:
                    actions.append(model.get_aileron_pwm(state))
                
                aileron_pwm = np.mean(actions)
                print("actions:", actions)

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
        self.action_history["pitch"].append(tuple([pwm, sign * saturated]))
        self.elevator_mutex.release()
    
        # print(f'\n---_elevator_control---\nthread name: {thread_name}\nsaturated: {saturated}')
                
    def elevator_function(self):
        P = 0.03
        D = 0.01
        with ThreadPoolExecutor(max_workers=self.elevator_worker_num, 
                                thread_name_prefix="Elevator") as elevator:
            while self.simulation_running:
                self.observation_event.wait()
                               
                data = self.flight_data_stack[-1][0]
                
                if self.waypoint_index < len(self.waypoints) - 2 or (self.waypoint_index >= len(self.waypoints) - 3 and self.distance_to_waypoint >= 1.4):
                    ver_vel = data[5]
                    elevator_pwm = ver_vel * P
                    
                    elevator.submit(self._elevator_control, elevator_pwm)
                else:
                    pitch = data[8]
                    pitch_rate = data[9]
                    LOS = data[12] # new pitch or LOS
                    
                    lt = LOS + 30 # - self.terminal_pitch_target
                    lp = LOS - pitch
                    pitch_err = lt + lp
                    
                    elevator_pwm =  pitch_err * P - pitch_rate * D
                    elevator.submit(self._elevator_control, elevator_pwm)
        
        print('elevator thread ended')
    
    def get_initial_postition(self):
        # return tuple([self.initial_pos_x, self.initial_pos_y, self.initial_pos_z])
        return tuple([self.initial_latitude, self.initial_longitude])
    
    def _bezier_curve(self, t, P0, P1, P2, P3):
        """Computes a point on a cubic Bézier curve for a given t."""
        x = (1 - t)**3 * P0[0] + 3 * (1 - t)**2 * t * P1[0] + 3 * (1 - t) * t**2 * P2[0] + t**3 * P3[0]
        y = (1 - t)**3 * P0[1] + 3 * (1 - t)**2 * t * P1[1] + 3 * (1 - t) * t**2 * P2[1] + t**3 * P3[1]
        return y, x
    
    def calculate_bezier_curve(self, initial_pos, target_pos):
        # (x: longitude, y: latitude)
        # Define control points
        P0 = initial_pos
        P1 = (initial_pos[0], initial_pos[1] + (target_pos[1] - initial_pos[1])*2/3)
        P2 = (target_pos[0], target_pos[1] - ((target_pos[1] - initial_pos[1])*2/3))
        P3 = target_pos

        # Generate curve points
        t_values = np.linspace(0, 1, 8)  # N points along the curve
        curve_points = [self._bezier_curve(t, P0, P1, P2, P3) for t in t_values]
        
        return curve_points[1:]
    
    def get_observation(self):
        print('observation thread started...\n')
        while self.simulation_running:
            try:
                if self.sock and not self.sock._closed:
                    d, a = self.sock.recvfrom(2048) # 2048 maximum bit to receive
                    
                values = readPacket(d) # extract the data received
                    
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
                
                # calculate the waypoints
                if self.target_latitude == None and self.target_longitude == None and abs(values.get('target_lat')) > 0.0 and abs(values.get('target_long')) > 0.0:
                    self.target_latitude = values.get('target_lat')
                    self.target_longitude = values.get('target_long')
                    
                    print(f'\ntarget_latitude: {self.target_latitude}\ntarget_longitude: {self.target_longitude}\n')
                    
                    if self.get_direction(yaw) in ['N', 'S'] and (not self.target_latitude == None) and (not self.target_longitude == None):
                        true_yaw = (yaw + 180) % 180
                        
                        lat_gap = math.sin(math.radians(90-true_yaw)) * STRAIGHT_DIST
                        long_gap = math.sin(math.radians(true_yaw)) * STRAIGHT_DIST
                        
                        diameter_lat_gap = math.sin(math.radians(true_yaw)) * DIAMETER_TURNING_POINT
                        diameter_long_gap = math.sin(math.radians(90-true_yaw)) * DIAMETER_TURNING_POINT
                        
                        if self.get_direction(yaw) == 'S':
                            self.straight_dist_lat = self.target_latitude - lat_gap     # (-5.258)
                            self.straight_dist_long = self.target_longitude + long_gap  # (-145.92)
                        
                            self.turning_point_lat = self.straight_dist_lat - diameter_lat_gap # (-5.23)
                            self.turning_point_long = self.straight_dist_long - diameter_long_gap # (-145.95)
                            print(f'\nSTRAIGHT_DIST: {STRAIGHT_DIST}\ntrue_yaw: {true_yaw}\nstraight_dist_lat: {self.straight_dist_lat}\nlat_gap: {lat_gap}\nstraight_dist_long: {self.straight_dist_long}\nlong_gap: {long_gap}\n\ndiameter_long_gap: {diameter_long_gap}\ndiameter_lat_gap: {diameter_lat_gap}\nturning_point_lat: {self.turning_point_lat}\nturning_point_long: {self.turning_point_long}\n')
                        
                        trajectory_points = self.calculate_bezier_curve((self.initial_longitude, self.initial_latitude), (self.turning_point_long, self.turning_point_lat))
                        print(f'trajectory_points: {trajectory_points}')
                        self.waypoints = trajectory_points + [(self.straight_dist_lat, self.straight_dist_long), (self.target_latitude, self.target_longitude)]
                        
                is_grounded = values.get('grounded') or values.get('destroyed')
                
                self.flight_data_stack.append([tuple([pos_x, pos_y, pos_z, altitude,
                                                      velocity, ver_vel, 
                                                      roll, roll_rate, 
                                                      pitch, pitch_rate, 
                                                      yaw, yaw_rate, 
                                                      new_pitch, new_yaw,
                                                      latitude, longitude]), 
                                                      is_grounded])
                
                # event set
                self.observation_event.set()
                self.observation_event.clear()
                
            except socket.timeout:
                print('\n[ERROR]: TimeoutError at get_observation method\n')
                                
                is_grounded = True
                self.flight_data_stack.append([tuple([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), is_grounded])
                
                # event set
                self.observation_event.set()
                self.observation_event.clear()
                
        print('observation thread ended...\n')
        self.observation_event.set()
    
    
    def get_flight_data_stack(self):
        return self.flight_data_stack
    
    def get_action(self, action_list, i):
        return action_list[i] if i < len(action_list) else (np.nan, np.nan)
        
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
        black_box_filename = f"flight_data/state_history_yaw180_{self.model_name}_{self.timestr}.xlsx"
        mode = 'w' if self.ep == 0 else 'a'
        with pd.ExcelWriter(black_box_filename, mode=mode) as writer:
            black_box_values.to_excel(writer, sheet_name=f"episode_{self.ep}")

        max_len = max(len(self.action_history["pitch"]),
                      len(self.action_history["roll"]),
                      len(self.action_history["yaw"]))
                
        action_history_combined = [
            (
                self.get_action(self.action_history["roll"], i)[0], self.get_action(self.action_history["roll"], i)[1],
                self.get_action(self.action_history["pitch"], i)[0], self.get_action(self.action_history["pitch"], i)[1],
                self.get_action(self.action_history["yaw"], i)[0], self.get_action(self.action_history["yaw"], i)[1],
            )
            for i in range(max_len)
        ]
        
        action_history_values = pd.DataFrame(
            action_history_combined,
            columns=['aileron_pwm', 'saturated_aileron_pwm',
                     'elevator_pwm', 'saturated_elevator_pwm',
                     'rudder_pwm', 'saturated_rudder_pwm']
        )

        action_history_filename = f"flight_data/action_history_yaw180_{self.model_name}_{self.timestr}.xlsx"
        mode = 'w' if self.ep == 0 else 'a'
        with pd.ExcelWriter(action_history_filename, mode=mode) as writer:
            action_history_values.to_excel(writer, sheet_name=f"episode_{self.ep}")    
    
        
        