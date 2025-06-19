# -*- coding: utf-8 -*-
"""
Created on Sat May 24 20:39:22 2025

@author: ammar
"""

from concurrent.futures import ThreadPoolExecutor
import threading, time, socket, random, math
from tqdm import trange
import pandas as pd
import numpy as np

from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
from wincam import DXCamera
# import win32gui
import cv2

from src.srvision import draw_distance, draw_crosshair, CENTER_X, CENTER_Y

from src.receiver import readPacket
from pyKey.pyKey_windows import pressKey, releaseKey, press
from src.utils import relaunch, unpause_game, switch_tab, switch_to_fpv, slow_sim
from elevator_controller.model_fix.nn_model import YoloElevator


class Simulation:
    def __init__(self, num_episode=1, max_step =1000, 
                 IP="localhost", main_port=2873,
                 show_fps=False,
                 yolo_model_path="D:\\Projects\\YOLOv8_training\\runs\\detect\\train17\\weights\\best.pt",
                 simpplerockets_screen=[0, 68, 1024, 768],
                 windows_name='YOLO-Targeting-Sim',
                 model_name='YOLOv8N',
                 yolo_elevator_model=YoloElevator(),
                 is_nn_model=True,
                 fps=30):
        
        # class variable
        self.num_ep = num_episode
        self.ep = 0
        self.max_step = max_step
        self.show_fps = show_fps
        self.delta_time = 0.2
        self.saturation = 1
        self.distance_to_target = '##'
        
        # communication properties            
        self.IP = IP
        self.main_port = main_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(30) # 30 seconds timeout
        self.sock.bind((self.IP, self.main_port))
        
        # for object detection
        self.yolo_model_path = yolo_model_path
        self.yolo_model = YOLO(self.yolo_model_path)
        self.simpplerockets_screen = simpplerockets_screen
        self.fps = fps
        self.fps_list = []
        self.windows_name = windows_name
        
        # fot elevator controller
        self.yolo_elevator_model = yolo_elevator_model
        self.is_nn_model = is_nn_model
        
        # global variables
        self.flight_data_stack = [[tuple([0]*14), False]]
        self.simulation_running = True
        self.action_history = {"pitch":[]}
        self.frame_list = []
        self.model_name = model_name
        self.target_center = None
        self.prev_target_center = None
        self.wh = None
        self.prev_wh = None
        
        # antiroll models
        # self.models = []
        # self.models.append(PID_Model(P=0.014, I=0.001, D=0.012))
        # self.models.append(LR_Model())
        # self.models.append(LSTM_Model())
        # self.models.append(DDPG_Model(model_path='controllers/DDPG/training_results/20241102-082455/weights/20241102-082455'))
        
        # thread properties
        self.elevator_worker_num = 3

        # synchronization objects
        self.observation_event = threading.Event()
        self.yolo_event = threading.Event()
        self.yolo_is_working_event = threading.Event()
        self.elevator_mutex = threading.Lock()
        self.elevator_threads_communication = {f"Elevator_{i}": threading.Event() for i in range(self.elevator_worker_num)}
        
        # threads initialization
        self.pitch_contoller = threading.Thread(target=self.elevator_function, args=(), name='pitch_contoller')
        self.yolo_pitch_controller = threading.Thread(target=self.yolo_elevator_function, args=(), name='yolo_pitch_contoller')
        self.observator = threading.Thread(target=self.get_observation, args=(), name='observator')
        self.object_detection_thread = threading.Thread(target=self.object_detection_function, args=(), name='target_detector')
        
    def run(self):
        switch_tab()
        time.sleep(0.5)
        switch_to_fpv()
        time.sleep(0.5)
        # self.object_detection_thread.start()
        time.sleep(1)
        # start the threads
        self.observator.start()
        self.yolo_is_working_event.set()
        self.pitch_contoller.start()
        # self.yolo_pitch_controller.start()
        unpause_game()

        loop_time = time.time()
        with trange(self.num_ep) as t:
            for ep in t:
                self.ep = ep
                self.reset()
                
                grounded = False
                
                while not grounded:
                    self.observation_event.wait()
                    grounded = self.flight_data_stack[-1][1]
                    
                    if self.show_fps:
                        print('\nFPS {}'.format(1 / (time.time() - loop_time)))
                        loop_time = time.time()
                                       
        
        time.sleep(3)
        self.set_simulation_running_false()
        print('main thread ended')
        self.save_video()
    
    def reset(self):
        releaseKey('w')
        releaseKey('s')
        releaseKey('a')
        releaseKey('d')
        releaseKey('q')
        releaseKey('e')
        if self.ep != 0:
            self.set_missile()
    
    def close(self):
        self.observation_event.set()
    
    def set_missile(self):
        relaunch(end_flight=[125, 170], retry_undo=[520, 627], retry=[517, 569])
    
    def set_simulation_running_false(self):
        self.simulation_running = False
    
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
    
        
# =============================================================================
# thread functions 
# =============================================================================    
    
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
    
    def yolo_elevator_function(self):
        with ThreadPoolExecutor(max_workers=self.elevator_worker_num, 
                                thread_name_prefix="Elevator") as elevator:
            while self.simulation_running:                
                self.yolo_is_working_event.set()
                self.yolo_event.wait()
                self.yolo_is_working_event.clear()
                
                distance = CENTER_Y - self.prev_target_center[1]
                wh = self.prev_wh
                
                elevator_pwm = 0
                if self.is_nn_model:
                    elevator_pwm = self.yolo_elevator_model.infer(input_vector=[distance] + wh)[0]
                else:
                    input_vector = pd.DataFrame([[distance] + wh], columns=["dy", "w", "h"])
                    elevator_pwm = self.yolo_elevator_model.predict(input_vector)[0]
                    
                print("elevator_pwm:", elevator_pwm)
                elevator.submit(self._elevator_control, elevator_pwm)
        
        print('elevator thread ended')
        
    def elevator_function(self):
        P = 0.03
        D = 0.01
        with ThreadPoolExecutor(max_workers=self.elevator_worker_num, 
                                thread_name_prefix="Elevator") as elevator:
            while self.simulation_running:
                self.observation_event.wait()
                
                # if yolo is working, it will wait
                self.yolo_is_working_event.wait()
                
                data = self.flight_data_stack[-1][0]
                pitch = data[8]
                pitch_rate = data[9]
                LOS = data[12] # new pitch or LOS
                
                lt = LOS + 30 # - self.terminal_pitch_target
                lp = LOS - pitch
                pitch_err = lt + lp
                
                elevator_pwm =  pitch_err * P - pitch_rate * D
                elevator.submit(self._elevator_control, elevator_pwm)
        
        print('elevator thread ended')
        
    def object_detection_function(self):
        loop_time = time.time()
        with DXCamera(self.simpplerockets_screen[0],
                      self.simpplerockets_screen[1],
                      self.simpplerockets_screen[2],
                      self.simpplerockets_screen[3], 
                      capture_cursor=False,
                      fps=self.fps) as camera:
            
            while self.simulation_running:
                self.target_center = None
                
                frame, timestamp = camera.get_bgr_frame()
                frame = np.ascontiguousarray(frame)
                # print('frame:', frame.shape)
                
                results = self.yolo_model(frame, verbose=False)
                
                annotator = Annotator(frame)
                boxes = results[0].boxes
                
                if len(boxes) > 0:
                    index = 0
                    max_conf = boxes[index].conf.item()
                    for i in range(1, len(boxes)):
                        temp_conf = boxes[i].conf.item()
                        if temp_conf > max_conf:
                            index = i
                            max_conf = temp_conf
                    
                    b = boxes[index].xyxy[0]
                    max_conf *= 100
                    try:
                        label = "{:.2f}".format(self.distance_to_target)
                    except:
                        label = "##"
                    annotator.box_label(b, label, color=(0, 0, 255))
                    
                    self.target_center = boxes[index].xywh[0].tolist()[:2]
                    self.wh = boxes[index].xywh[0].tolist()[2:]
                    
                    # keep the previous target center coordinate
                    self.prev_target_center = self.target_center
                    self.prev_wh = self.wh
                    
                    # wake up the yolo elevator controller
                    self.yolo_event.set()
                    self.yolo_event.clear()
                
                else:
                    self.target_center = None
                    
                    if self.prev_target_center == None:
                        self.prev_target_center = (CENTER_X, CENTER_Y)   
                        self.prev_wh = [0, 0]
                    
                screenshot = annotator.result()
                
                image = draw_crosshair(screenshot)
                image = draw_distance(image, self.target_center)
                # print("target_center:", self.target_center)
                
                # Save frame to list
                self.frame_list.append(image.copy())
                
                # display_screenshot = cv2.resize(image, (676, 507))
                
                # cv2.imshow(self.windows_name, display_screenshot)
                # cv2.moveWindow(self.windows_name, 1030, 31)
                
                # if self.simulation_running == "Running":
                #     hWndYOLO = win32gui.FindWindow(None, self.windows_name)
                #     win32gui.SetForegroundWindow(hWndYOLO)
                #     hWnd = win32gui.FindWindow(None, 'Juno: New Origins')
                #     win32gui.SetForegroundWindow(hWnd)
                # if self.simulation_running == "Ended":
                #     hWnd = win32gui.FindWindow(None, self.windows_name)
                #     win32gui.SetForegroundWindow(hWnd)
                #     press('q')
                
                # print('FPS {}'.format(1 / (time.time() - loop_time)))
                
                fps = 1 / (time.time() - loop_time)
                self.fps_list.append(fps)
                
                loop_time = time.time()
                
    def save_video(self):
        # Save video after loop ends
        output_path = "output_video.mp4"
        if self.frame_list:
            height, width, _ = self.frame_list[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = sum(self.fps_list) / len(self.fps_list)
            print("AVERAGE FPS:", fps)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame in self.frame_list:
                out.write(frame)
            out.release()
            print(f"Video saved to {output_path}")
        
        cv2.destroyAllWindows()
    
    def get_observation(self):
        print('observation thread started...\n')
        while self.simulation_running:
            try:
                if self.sock and not self.sock._closed:
                    d, a = self.sock.recvfrom(2048) # 2048 maximum bit to receive
                    
                values = readPacket(d) # extract the data received
                
                self.distance_to_target = values.get('distance_to_target')
                    
                pos_x = values.get('pos_x')
                pos_y = values.get('pos_y')
                pos_z = values.get('pos_z')
                
                latitude = values.get('latitude')
                longitude = values.get('longitude')

                altitude = values.get('agl')
                velocity = values.get('velocity')
                ver_vel = values.get('ver_vel')
        
                roll = values.get('roll')
                roll_rate = values.get('roll_rate')


                pitch = values.get('pitch')
                pitch_rate = values.get('pitch_rate')
                new_pitch = values.get('new_pitch')
                
                if self.distance_to_target <= 650:
                    # Add noise to pitch with 50% probability
                    if random.random() < 0.5:
                        noise = random.uniform(-5, 5)
                        pitch += noise
                
                    # Add noise to pitch_rate with 50% probability
                    if random.random() < 0.5:
                        noise = random.uniform(-2, 2)
                        pitch_rate += noise
                
                    # Add noise to new_pitch with 50% probability
                    if random.random() < 0.5:
                        noise = random.uniform(-5, 5)
                        new_pitch += noise
                
                yaw = values.get('yaw') if values.get('yaw') <= 180 else (values.get('yaw') - 360)
                yaw_rate = values.get('yaw_rate')
                
                new_yaw = values.get('new_yaw') # this is new heading or the line of sight for the yaw to target.
                
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
        black_box_filename = f"flight_data/state_history_yolo_{self.model_name}_{self.timestr}.xlsx"
        mode = 'w' if self.ep == 0 else 'a'
        with pd.ExcelWriter(black_box_filename, mode=mode) as writer:
            black_box_values.to_excel(writer, sheet_name=f"episode_{self.ep}")
                
        action_history_combined = [
            (
                self.get_action(self.action_history["pitch"], i)[0], self.get_action(self.action_history["pitch"], i)[1],
            )
            for i in range(len(self.action_history["pitch"]))
        ]
        
        action_history_values = pd.DataFrame(
            action_history_combined,
            columns=['elevator_pwm', 'saturated_elevator_pwm']
        )

        action_history_filename = f"flight_data/action_history_yolo_{self.model_name}_{self.timestr}.xlsx"
        mode = 'w' if self.ep == 0 else 'a'
        with pd.ExcelWriter(action_history_filename, mode=mode) as writer:
            action_history_values.to_excel(writer, sheet_name=f"episode_{self.ep}")    
    

