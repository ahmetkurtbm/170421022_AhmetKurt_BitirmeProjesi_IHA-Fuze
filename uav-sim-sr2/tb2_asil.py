import socket, struct, threading, time
from pyKey_windows import pressKey, releaseKey
from concurrent.futures import ThreadPoolExecutor
from utils import unpause_game, switch_tab

TypeFormats = [
    "",     # null
    "d",    # double (float64)
    "?",    # bool
    "ddd"]  # Vector3d

class Packet:
    def __init__(self, data):
        self.data = data
        self.pos = 0
    def get(self, l): # get l bytes
        self.pos += l
        return self.data[self.pos - l:self.pos]
    def read(self, fmt): #  read all formatted values
        v = self.readmult(fmt)
        if len(v) == 0: return None
        if len(v) == 1: return v[0]
        return v
    def readmult(self, fmt): # read multiple formatted values
        return struct.unpack(fmt, self.get(struct.calcsize(fmt)))
    @property
    def more(self): # is there more data?
        return self.pos < len(self.data)

def readPacket(dat):
    p = Packet(dat)
    
    values = {}
    while p.more:
        nameLen = p.read("H")
        name = p.get(nameLen).decode()
        
        tp = p.read("B")
        typeCode = TypeFormats[tp]
        val = p.read(typeCode)
        
        values[name] = val
        # print(name, "=", val)

    return values

class PID_Aileron:
    
    def __init__(self, P=0.016, I=0.001, D=0.009):
        self.P = P
        self.I = I
        self.D = D
        
        self.roll = 0
        self.roll_rate = 0
        self.accumulated_roll_error = 0
        
    def get_aileron_pwm(self, roll, roll_rate):
        self.roll = roll
        self.roll_rate = roll_rate
        
        self.roll_error = -self.roll
        self.accumulated_roll_error += self.roll_error
        self.aileron_pwm = (self.roll_error * self.P + 
                            self.accumulated_roll_error * self.I + 
                            (self.roll_rate + 0.000001) * self.D) * 0.5
        
        return self.aileron_pwm
    
# constants
NUM_STATE = 11

class TB2:
    def __init__(self, IP="localhost", main_port=2873, display_data=False):
        # communication properties            
        self.IP = IP
        self.main_port = main_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(30) # 30 seconds timeout
        self.sock.bind((self.IP, self.main_port))

        self.simulation_running = True
        self.stabilize_pitch = True
        self.stabilize_roll = True

        # initial position
        self.intial_pos_defined = False
        self.initial_latitude = None
        self.initial_longitude = None
        self.flight_data_stack = [[tuple([0]*NUM_STATE), False]]

        # multithreading
        self.data_receiver = threading.Thread(target=self.get_flight_data, args=(), name='data_receiver')
        self.flight_data_event = threading.Event()
        self.pitch_stabilizer = threading.Thread(target=self.pitch_stabilizer_function, args=(), name='pitch_stabilizer')
        self.roll_stabilizer = threading.Thread(target=self.roll_stabilizer_function, args=(), name='roll_stabilizer')
        self.num_pitch_worker = 3
        self.mutex_pitch_stabilizer = threading.Lock()
        self.pitch_threads_communication = {f"Pitch_{i}": threading.Event() for i in range(self.num_pitch_worker)}
        self.num_roll_worker = 3
        self.mutex_roll_stabilizer = threading.Lock()
        self.roll_threads_communication = {f"Roll_{i}": threading.Event() for i in range(self.num_roll_worker)}
        
        # PID stabilizer
        self.pid_aileron = PID_Aileron()

        self.display_data = display_data
        self.delta_time = 0.2
        self.saturation = 1

    def run(self):
        switch_tab()
        time.sleep(0.2)

        # start the threads
        self.data_receiver.start()
        self.pitch_stabilizer.start()
        self.roll_stabilizer.start()

        unpause_game()

        grounded = False
        while not grounded:
            self.flight_data_event.wait()
            grounded = self.flight_data_stack[-1][1]
            # print("flight_data_stack: ", self.flight_data_stack[-1])

        self.close()
        print('Main thread ended')

    def close(self):
        self.simulation_running = False
        self.flight_data_event.set()

    def get_flight_data(self):
        print('flight data receiver thread started...\n')
        while self.simulation_running:
            try:
                if self.sock and not self.sock._closed:
                    d, a = self.sock.recvfrom(2048) # 2048 maximum bit to receive
                    
                values = readPacket(d) # extract the data received
                    
                altitude = values.get('agl')
                latitude = values.get('latitude')
                longitude = values.get('longitude')               
                velocity = values.get('velocity')
                ver_vel = values.get('ver_vel')
                roll = values.get('roll')
                roll_rate = values.get('roll_rate')
                pitch = values.get('pitch')
                pitch_rate = values.get('pitch_rate')
                yaw = values.get('yaw') if values.get('yaw') <= 180 else (values.get('yaw') - 360)
                yaw_rate = values.get('yaw_rate')

                is_grounded = values.get('grounded') or values.get('destroyed')

                # set the initial position
                if not self.intial_pos_defined:                    
                    self.initial_latitude = latitude
                    self.initial_longitude = longitude
                    self.intial_pos_defined = True

                self.flight_data_stack.append([tuple([altitude, latitude, longitude,
                                                      velocity, ver_vel, 
                                                      roll, roll_rate, 
                                                      pitch, pitch_rate, 
                                                      yaw, yaw_rate, 
                                                      ]), 
                                                      is_grounded])
                
                if self.display_data:
                    print(f"\nAltitude: {altitude:.2f}, Latitude: {latitude:.2f}, Longitude: {longitude:.2f}, "
                        f"Velocity: {velocity:.2f}, Vertical Velocity: {ver_vel:.2f}, "
                        f"Roll: {roll:.2f}, Roll Rate: {roll_rate:.2f}, "
                        f"Pitch: {pitch:.2f}, Pitch Rate: {pitch_rate:.2f}, "
                        f"Yaw: {yaw:.2f}, Yaw Rate: {yaw_rate:.2f}")

            except socket.timeout:
                print('\n[ERROR]: TimeoutError at get_flight_data method\n')
                                
                is_grounded = True
                self.flight_data_stack.append([(0,) * NUM_STATE, is_grounded])
            finally:
                # event set
                self.flight_data_event.set()
                self.flight_data_event.clear()

    def _elevator_control(self, pwm):
        thread_name = threading.current_thread().name

        self.mutex_pitch_stabilizer.acquire()
        self.pitch_threads_communication[thread_name].set()  # Signal thread communication active
        self.mutex_pitch_stabilizer.release()

            
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
    
        self.mutex_pitch_stabilizer.acquire()
        self.pitch_threads_communication[thread_name].clear()  # Signal thread communication complete
        if not any(event.is_set() for event in self.pitch_threads_communication.values()):
            releaseKey('w')
            releaseKey('s')
        self.mutex_pitch_stabilizer.release()

    def _aileron_control(self, pwm):
        thread_name = threading.current_thread().name
    
        self.mutex_roll_stabilizer.acquire()
        self.roll_threads_communication[thread_name].set()  # Signal thread communication active
        self.mutex_roll_stabilizer.release()
    
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
    
        self.mutex_roll_stabilizer.acquire()
        self.roll_threads_communication[thread_name].clear()  # Signal thread communication complete
        if not any(event.is_set() for event in self.roll_threads_communication.values()):
            releaseKey('q')
            releaseKey('e')
        self.mutex_roll_stabilizer.release()

    def pitch_stabilizer_function(self, P=0.008):
        with ThreadPoolExecutor(max_workers=self.num_pitch_worker, 
                                thread_name_prefix="Pitch") as stabilizer:
            while self.simulation_running:
                self.flight_data_event.wait()
                if self.stabilize_pitch:
                    data = self.flight_data_stack[-1][0]
                    ver_vel = data[4]
                    elevator_pwm = ver_vel * P
                    stabilizer.submit(self._elevator_control, elevator_pwm)

    def roll_stabilizer_function(self):
        with ThreadPoolExecutor(max_workers=self.num_roll_worker, 
                                thread_name_prefix="Roll") as stabilizer:
            while self.simulation_running:
                self.flight_data_event.wait()
                if self.stabilize_roll:
                    data = self.flight_data_stack[-1][0]
                    roll = data[5]
                    roll_rate = data[6]
                    aileron_pwm = self.pid_aileron.get_aileron_pwm(roll, roll_rate)
                    stabilizer.submit(self._aileron_control, aileron_pwm)

    def set_stabilize_pitch(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Expected a boolean")
        self.stabilize_pitch = value

    def set_stabilize_roll(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Expected a boolean")
        self.stabilize_roll = value



    