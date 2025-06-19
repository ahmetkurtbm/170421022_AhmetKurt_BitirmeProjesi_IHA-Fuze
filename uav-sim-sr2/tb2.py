import socket, struct, threading, time
import math
from pyKey_windows import pressKey, releaseKey
from concurrent.futures import ThreadPoolExecutor
from utils import unpause_game, switch_tab
import pandas as pd
from datetime import datetime
import os

# Veri türü formatları
TypeFormats = [
    "",     # boş
    "d",    # double (float64)
    "?",    # bool
    "ddd"]  # Vector3d

class Packet:
    """Veri paketi okuma sınıfı"""
    def __init__(self, data):
        self.data = data
        self.pos = 0
        
    def get(self, l): 
        """l kadar byte al"""
        self.pos += l
        return self.data[self.pos - l:self.pos]
        
    def read(self, fmt): 
        """Formatlanmış değerleri oku"""
        v = self.readmult(fmt)
        if len(v) == 0: return None
        if len(v) == 1: return v[0]
        return v
        
    def readmult(self, fmt): 
        """Çoklu formatlanmış değerleri oku"""
        return struct.unpack(fmt, self.get(struct.calcsize(fmt)))
        
    @property
    def more(self): 
        """Daha fazla veri var mı?"""
        return self.pos < len(self.data)

def readPacket(dat):
    """Veri paketini oku ve sözlük olarak döndür"""
    p = Packet(dat)
    
    values = {}
    while p.more:
        nameLen = p.read("H")
        name = p.get(nameLen).decode()
        
        tp = p.read("B")
        typeCode = TypeFormats[tp]
        val = p.read(typeCode)
        
        values[name] = val

    return values

class Waypoint:
    """Waypoint sınıfı - her waypoint için konum ve tolerans"""
    def __init__(self, latitude, longitude, altitude=None, tolerance=0.0001):
        self.latitude = latitude         # Enlem
        self.longitude = longitude       # Boylam
        self.altitude = altitude         # Yükseklik
        self.tolerance = tolerance       # Waypoint'e ne kadar yakın sayılacağı
        self.reached = False            # Ulaşıldı mı?

class WaypointNavigator:
    """Waypoint navigasyon sistemi"""
    def __init__(self):
        self.waypoints = []                    # Waypoint listesi
        self.current_waypoint_index = 0        # Aktif waypoint indeksi
        self.navigation_active = False         # Navigasyon aktif mi?
        
    def add_waypoint(self, waypoint):
        """Yeni waypoint ekle"""
        self.waypoints.append(waypoint)
        
    def clear_waypoints(self):
        """Tüm waypoint'leri temizle"""
        self.waypoints.clear()
        self.current_waypoint_index = 0
        
    def get_current_waypoint(self):
        """Aktif waypoint'i döndür"""
        if self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        return None
        
    def check_waypoint_reached(self, current_lat, current_lon, current_alt=None):
        """Waypoint'e ulaşılıp ulaşılmadığını kontrol et"""
        current_wp = self.get_current_waypoint()
        if current_wp is None:
            return False
            
        # Mesafe hesaplama (haversine formülü)
        distance = self.calculate_distance(current_lat, current_lon, 
                                         current_wp.latitude, current_wp.longitude)
        
        # Waypoint'e ulaşıldı mı kontrol et
        if distance < current_wp.tolerance:
            current_wp.reached = True
            self.current_waypoint_index += 1
            print(f"Waypoint {self.current_waypoint_index} hedefine ulaşıldı!")
            return True
        return False
        
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """İki nokta arası mesafe hesaplama (metre cinsinden)"""
        R = 1274200  # Dünya yarıçapı (metre)
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c  # metre cinsinden
        
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """İki nokta arası yön hesaplama (derece cinsinden)"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360  # 0-360 arası normalize et
        
        return bearing

class PID_Aileron:
    """Roll (yalpa) kontrolü için PID kontrolcüsü"""
    def __init__(self, P=0.01, I=0.0002, D=0.005):  # PID değerlerini düşürdüm
        self.P = P  # Orantısal kazanç
        self.I = I  # İntegral kazanç
        self.D = D  # Türevsel kazanç
        
        self.roll = 0                      # Mevcut roll değeri
        self.roll_rate = 0                 # Roll değişim hızı
        self.accumulated_roll_error = 0    # Birikmiş roll hatası
        self.max_integral = 10            # İntegral sınırını düşürdüm
        self.last_error = 0               # Son hata değeri
        self.last_time = time.time()      # Son hesaplama zamanı
        self.error_history = []           # Hata geçmişi
        self.approach_mode = False        # Yaklaşma modu
        
    def set_approach_mode(self, mode):
        """Yaklaşma modunu ayarla"""
        self.approach_mode = mode
        if mode:
            # Yaklaşma modunda integrali sıfırla
            self.accumulated_roll_error = 0
        
    def get_aileron_pwm(self, roll, roll_rate, target_roll=0):
        """Aileron PWM değerini hesapla"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        self.roll = roll
        self.roll_rate = roll_rate
        
        # Hedef roll ile mevcut roll arasındaki fark
        self.roll_error = target_roll - self.roll
        
        # Hata geçmişini güncelle
        self.error_history.append(self.roll_error)
        if len(self.error_history) > 5:
            self.error_history.pop(0)
            
        # İntegral sınırlaması
        if not self.approach_mode:  # Sadece normal modda integral kullan
            self.accumulated_roll_error = max(-self.max_integral, 
                                            min(self.max_integral, 
                                                self.accumulated_roll_error + self.roll_error * dt))
        
        # Türevsel terim için hata değişimi
        error_derivative = (self.roll_error - self.last_error) / dt if dt > 0 else 0
        self.last_error = self.roll_error
        
        # PID hesaplaması - yaklaşma modunda daha hassas kontrol
        if self.approach_mode:
            self.aileron_pwm = (self.roll_error * self.P * 0.8 +  # Yaklaşma modunda daha yumuşak
                               error_derivative * self.D * 0.5) * 0.6
        else:
            self.aileron_pwm = (self.roll_error * self.P + 
                               self.accumulated_roll_error * self.I + 
                               error_derivative * self.D) * 0.7
        
        # PWM değerini sınırla
        self.aileron_pwm = max(-1.0, min(1.0, self.aileron_pwm))
        
        return self.aileron_pwm

class PID_Elevator:
    """Pitch (yükseliş) kontrolü için PID kontrolcüsü"""
    def __init__(self, P=0.0001, I=0.0001, D=0.0003):  # PID değerlerini düşürdüm
        self.P = P  # Orantısal kazanç
        self.I = I  # İntegral kazanç
        self.D = D  # Türevsel kazanç
        
        self.pitch = 0                    # Mevcut pitch değeri
        self.pitch_rate = 0               # Pitch değişim hızı
        self.accumulated_pitch_error = 0  # Birikmiş pitch hatası
        self.max_integral = 8            # İntegral sınırını düşürdüm
        self.last_error = 0              # Son hata değeri
        self.last_time = time.time()     # Son hesaplama zamanı
        self.error_history = []          # Hata geçmişi

class NavigationController:
    """Navigasyon kontrolcüsü - Pitch ve Roll ile yönelim kontrolü"""
    def __init__(self, roll_P=0.8, pitch_P=0.02):  # Roll ve pitch değerlerini düşürdüm
        self.roll_P = roll_P
        self.pitch_P = pitch_P
        self.last_yaw_error = 0
        self.yaw_error_history = []
        self.approach_phase = False
        self.last_distance = float('inf')
        self.distance_increasing = False
        
    def get_navigation_commands(self, current_lat, current_lon, current_yaw, target_lat, target_lon):
        target_bearing = self.calculate_bearing(current_lat, current_lon, target_lat, target_lon)
        yaw_error = target_bearing - current_yaw
        # Normalize yaw error to [-180, 180]
        yaw_error = (yaw_error + 180) % 360 - 180
        
        # Mesafe hesapla
        distance = self.calculate_distance(current_lat, current_lon, target_lat, target_lon)
        
        # Mesafe değişimini kontrol et
        if distance > self.last_distance:
            self.distance_increasing = True
        else:
            self.distance_increasing = False
        self.last_distance = distance
        
        # Waypoint'e yaklaşma fazını kontrol et
        if distance < 1000 and not self.approach_phase:
            self.approach_phase = True
        elif distance > 1500:
            self.approach_phase = False
            
        # Roll komutu: yaklaşma fazında daha hassas kontrol
        if self.approach_phase:
            if abs(yaw_error) > 5:
                if self.distance_increasing:
                    # Mesafe artıyorsa daha agresif dön
                    target_roll = yaw_error * self.roll_P * 2.0
                else:
                    # Mesafe azalıyorsa daha yumuşak dön
                    target_roll = yaw_error * self.roll_P
            else:
                target_roll = 0
        else:
            # Normal uçuş fazı
            if abs(yaw_error) > 10:
                target_roll = yaw_error * self.roll_P * 1.5
            else:
                target_roll = 0
                
        # Roll açısını sınırla
        target_roll = max(-30, min(30, target_roll))

        # Pitch komutu: yaklaşma fazında daha hassas kontrol
        if self.approach_phase:
            if distance < 500:
                target_pitch = 0  # Waypoint'e yakınken yükseklik koru
            else:
                target_pitch = -1  # Hafif alçal
        else:
            if abs(yaw_error) < 45:
                if distance > 1000:
                    target_pitch = -2
                else:
                    target_pitch = -1
            else:
                target_pitch = 1  # Dönüş sırasında yüksel

        print(f"DEBUG: Hedef yön: {target_bearing:.1f}°, Mevcut yön: {current_yaw:.1f}°, Yön hatası: {yaw_error:.1f}°")
        print(f"DEBUG: Hedef roll: {target_roll:.1f}°, Hedef pitch: {target_pitch:.1f}°, Mesafe: {distance:.1f}m")
        

        return target_roll, target_pitch
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """İki nokta arası mesafe hesaplama (metre)"""
        R = 6371000  # Dünya yarıçapı (metre)
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
        
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """İki nokta arası yön hesaplama (derece)"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing

# Sabitler
NUM_STATE = 11

class TB2:
    """TB2 İHA kontrol sınıfı"""
    def __init__(self, IP="localhost", main_port=2873, display_data=False):
        # İletişim özellikleri            
        self.IP = IP
        self.main_port = main_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(30)  # 30 saniye timeout
        self.sock.bind((self.IP, self.main_port))

        # Simülasyon durumu
        self.simulation_running = True
        self.stabilize_pitch = True    # Pitch stabilizasyonu
        self.stabilize_roll = True     # Roll stabilizasyonu
        
        # Waypoint navigasyon sistemi
        self.waypoint_navigator = WaypointNavigator()
        self.navigation_active = False
        self.navigation_controller = NavigationController()

        # İlk konum
        self.intial_pos_defined = False
        self.initial_latitude = None
        self.initial_longitude = None
        self.flight_data_stack = [[tuple([0]*NUM_STATE), False]]

        # Çoklu thread yapısı
        self.data_receiver = threading.Thread(target=self.get_flight_data, args=(), name='data_receiver')
        self.flight_data_event = threading.Event()
        self.pitch_stabilizer = threading.Thread(target=self.pitch_stabilizer_function, args=(), name='pitch_stabilizer')
        self.roll_stabilizer = threading.Thread(target=self.roll_stabilizer_function, args=(), name='roll_stabilizer')
        
        # Thread iletişimi için mutex ve event'ler
        self.num_pitch_worker = 3
        self.mutex_pitch_stabilizer = threading.Lock()
        self.pitch_threads_communication = {f"Pitch_{i}": threading.Event() for i in range(self.num_pitch_worker)}
        
        self.num_roll_worker = 3
        self.mutex_roll_stabilizer = threading.Lock()
        self.roll_threads_communication = {f"Roll_{i}": threading.Event() for i in range(self.num_roll_worker)}
        
        # PID kontrolcüleri
        self.pid_aileron = PID_Aileron()
        self.pid_elevator = PID_Elevator()

        # Diğer ayarlar
        self.display_data = display_data
        self.delta_time = 0.2
        self.saturation = 1
        self.approach_distance = 1000  # Yaklaşma mesafesi
        self.current_distance = float('inf')

        # Veri kayıt ayarları
        self.record_data = True
        self.flight_data_records = []
        self.start_time = None
        
        # Veri kayıt klasörlerini oluştur
        os.makedirs('flight_logs', exist_ok=True)

    def add_waypoint(self, latitude, longitude, altitude=None, tolerance=0.0001):
        """Waypoint ekle"""
        waypoint = Waypoint(latitude, longitude, altitude, tolerance)
        self.waypoint_navigator.add_waypoint(waypoint)
        print(f"Waypoint eklendi: Enlem {latitude}, Boylam {longitude}")
        
    def start_navigation(self):
        """Waypoint navigasyonunu başlat"""
        if len(self.waypoint_navigator.waypoints) > 0:
            self.navigation_active = True
            print("Navigasyon başlatıldı!")
        else:
            print("Hiç waypoint tanımlanmamış!")
            
    def stop_navigation(self):
        """Waypoint navigasyonunu durdur"""
        self.navigation_active = False
        print("Navigasyon durduruldu!")
        
    def clear_waypoints(self):
        """Tüm waypoint'leri temizle"""
        self.waypoint_navigator.clear_waypoints()
        print("Tüm waypoint'ler temizlendi!")

    def run(self):
        """Ana çalıştırma fonksiyonu"""
        switch_tab()
        time.sleep(0.2)

        # Thread'leri başlat
        self.data_receiver.start()
        self.pitch_stabilizer.start()
        self.roll_stabilizer.start()

        unpause_game()

        # İHA yere inene kadar bekle
        grounded = False
        while not grounded:
            self.flight_data_event.wait()
            grounded = self.flight_data_stack[-1][1]
            time.sleep(0.1)  # Her döngüde 100ms bekleme ekledim

        # Uçuş tamamlandığında verileri kaydet
        if self.record_data:
            self.save_flight_data()
        
        self.close()
        print('Ana thread sonlandı')

    def close(self):
        """Sistemi kapat"""
        self.simulation_running = False
        self.flight_data_event.set()

    def get_flight_data(self):
        """Uçuş verilerini al"""
        print('Uçuş veri alıcı thread başlatıldı...\n')
        while self.simulation_running:
            try:
                if self.sock and not self.sock._closed:
                    d, a = self.sock.recvfrom(2048)
                    
                values = readPacket(d)
                    
                # Uçuş verilerini çıkar
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

                # İlk konumu ayarla
                if not self.intial_pos_defined:                    
                    self.initial_latitude = latitude
                    self.initial_longitude = longitude
                    self.intial_pos_defined = True

                # Waypoint kontrolü
                if self.navigation_active:
                    current_wp = self.waypoint_navigator.get_current_waypoint()
                    if current_wp:
                        # Mesafeyi hesapla
                        self.current_distance = self.navigation_controller.calculate_distance(
                            latitude, longitude, current_wp.latitude, current_wp.longitude)
                        
                        # Yaklaşma modunu güncelle
                        self.pid_aileron.set_approach_mode(self.current_distance < self.approach_distance)
                        
                        # Waypoint'e ulaşıldı mı kontrol et
                        waypoint_reached = self.waypoint_navigator.check_waypoint_reached(
                            latitude, longitude, altitude)
                        
                        # Tüm waypoint'ler tamamlandıysa navigasyonu durdur
                        if self.waypoint_navigator.current_waypoint_index >= len(self.waypoint_navigator.waypoints):
                            print("Tüm waypoint'lere ulaşıldı! Navigasyon tamamlandı.")
                            self.navigation_active = False

                # Uçuş verilerini stack'e ekle
                self.flight_data_stack.append([tuple([altitude, latitude, longitude,
                                                      velocity, ver_vel, 
                                                      roll, roll_rate, 
                                                      pitch, pitch_rate, 
                                                      yaw, yaw_rate, 
                                                      ]), 
                                                      is_grounded])
                
                # Veri kaydı için, is_grounded = False olduğunda
                if self.record_data and not is_grounded:
                    if self.start_time is None:
                        self.start_time = datetime.now()
                    
                    elapsed_time = (datetime.now() - self.start_time).total_seconds()
                    
                    # Kaydedilecek veri dizisi
                    record = {
                        'time': elapsed_time,
                        'altitude': altitude,
                        'latitude': latitude,
                        'longitude': longitude,
                        'velocity': velocity,
                        'vertical_velocity': ver_vel,
                        'roll': roll,
                        'pitch': pitch,
                        'yaw': yaw,
                        'roll_rate': roll_rate,
                        'pitch_rate': pitch_rate,
                        'yaw_rate': yaw_rate
                    }
                    
                    # Eğer navigasyon aktifse, hedef bilgilerini de ekle
                    if self.navigation_active:
                        current_wp = self.waypoint_navigator.get_current_waypoint()
                        if current_wp:
                            record['target_latitude'] = current_wp.latitude
                            record['target_longitude'] = current_wp.longitude
                            record['distance_to_target'] = self.current_distance
                            record['waypoint_index'] = self.waypoint_navigator.current_waypoint_index + 1
                            record['total_waypoints'] = len(self.waypoint_navigator.waypoints)
                    
                    # Kayıtları listeye ekle
                    self.flight_data_records.append(record)

                # Veri görüntüleme
                if self.display_data:
                    current_wp = self.waypoint_navigator.get_current_waypoint()
                    if current_wp:
                        wp_info = f"Hedef: {self.waypoint_navigator.current_waypoint_index+1}/{len(self.waypoint_navigator.waypoints)}, Mesafe: {self.current_distance:.1f}m"
                    else:
                        wp_info = "Hedef yok"
                    
                    print(f"\nYükseklik: {altitude:.2f}m, Enlem: {latitude:.6f}, Boylam: {longitude:.6f}")
                    print(f"Hız: {velocity:.2f}m/s, Dikey Hız: {ver_vel:.2f}m/s")
                    print(f"Roll: {roll:.2f}°, Pitch: {pitch:.2f}°, Yaw: {yaw:.2f}°")
                    if self.navigation_active:
                        print(f"Navigasyon: {wp_info}")

            except socket.timeout:
                print('\n[HATA]: get_flight_data metodunda TimeoutError\n')
                                
                is_grounded = True
                self.flight_data_stack.append([(0,) * NUM_STATE, is_grounded])
            finally:
                self.flight_data_event.set()
                self.flight_data_event.clear()

    def save_flight_data(self):
        """Uçuş verilerini Excel dosyasına kaydet"""
        if not self.flight_data_records:
            print("[LOG] Kaydedilecek uçuş verisi bulunamadı.")
            return
            
        # DataFrame oluştur
        df = pd.DataFrame(self.flight_data_records)
        
        # Tarih ve saat formatı ile dosya adı oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flight_logs/tb2_flight_data_{timestamp}.xlsx"
        
        try:
            # Excel dosyasına kaydet
            df.to_excel(filename, index=False)
            print(f"[LOG] Uçuş verileri kaydedildi: {filename}")
            
            # İstatistikler
            flight_duration = df['time'].max() - df['time'].min()
            avg_altitude = df['altitude'].mean()
            max_velocity = df['velocity'].max()
            
            print(f"[STATS] Uçuş süresi: {flight_duration:.2f} saniye")
            print(f"[STATS] Ortalama yükseklik: {avg_altitude:.2f} m")
            print(f"[STATS] Maksimum hız: {max_velocity:.2f} m/s")
            
            # İstatistikleri kaydet
            stats_df = pd.DataFrame([{
                'flight_duration': flight_duration,
                'avg_altitude': avg_altitude,
                'max_velocity': max_velocity,
                'min_altitude': df['altitude'].min(),
                'max_altitude': df['altitude'].max(),
                'avg_velocity': df['velocity'].mean(),
                'total_waypoints': df['total_waypoints'].max() if 'total_waypoints' in df.columns else 0,
                'completed_waypoints': df['waypoint_index'].max() if 'waypoint_index' in df.columns else 0
            }])
            
            # İstatistikleri ayrı bir sayfaya kaydet
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                
        except Exception as e:
            print(f"[ERROR] Excel dosyası oluşturulurken hata: {e}")

    def _elevator_control(self, pwm):
        """Elevator (pitch) kontrolü"""
        thread_name = threading.current_thread().name

        self.mutex_pitch_stabilizer.acquire()
        self.pitch_threads_communication[thread_name].set()
        self.mutex_pitch_stabilizer.release()

        if pwm > 0:
            saturated = abs(min(self.delta_time * self.saturation, pwm))
            pressKey('s')  # Burun aşağı
            time.sleep(saturated)
        elif pwm < 0:
            saturated = abs(max(-(self.delta_time * self.saturation), pwm))
            pressKey('w')  # Burun yukarı
            time.sleep(saturated)

        self.mutex_pitch_stabilizer.acquire()
        self.pitch_threads_communication[thread_name].clear()
        if not any(event.is_set() for event in self.pitch_threads_communication.values()):
            releaseKey('w')
            releaseKey('s')
        self.mutex_pitch_stabilizer.release()

    def _aileron_control(self, pwm):
        """Aileron (roll) kontrolü"""
        thread_name = threading.current_thread().name
    
        self.mutex_roll_stabilizer.acquire()
        self.roll_threads_communication[thread_name].set()
        self.mutex_roll_stabilizer.release()
    
        if pwm > 0:
            saturated = abs(min(self.delta_time * self.saturation, pwm))
            pressKey('e')  # Sağa roll
            time.sleep(saturated)
        elif pwm < 0:
            saturated = abs(max(-(self.delta_time * self.saturation), pwm))
            pressKey('q')  # Sola roll
            time.sleep(saturated)

        self.mutex_roll_stabilizer.acquire()
        self.roll_threads_communication[thread_name].clear()
        if not any(event.is_set() for event in self.roll_threads_communication.values()):
            releaseKey('q')
            releaseKey('e')
        self.mutex_roll_stabilizer.release()

    def pitch_stabilizer_function(self):
        """Pitch stabilizasyon fonksiyonu"""
        with ThreadPoolExecutor(max_workers=self.num_pitch_worker, 
                                thread_name_prefix="Pitch") as stabilizer:
            while self.simulation_running:
                self.flight_data_event.wait()
                if self.stabilize_pitch:
                    data = self.flight_data_stack[-1][0]
                    
                    # Navigasyon aktifse hedef pitch'i hesapla
                    target_pitch = 0
                    # pitch_stabilizer_function içinde
                    if self.navigation_active:
                        current_wp = self.waypoint_navigator.get_current_waypoint()
                        if current_wp:
                            current_lat = data[1]
                            current_lon = data[2]
                            current_yaw = data[9]
                            target_roll, target_pitch = self.navigation_controller.get_navigation_commands(
                                current_lat, current_lon, current_yaw,
                                current_wp.latitude, current_wp.longitude)
                    else:
                        target_pitch = 0
                    
                    # Pitch kontrolü
                    current_pitch = data[7]
                    ver_vel = data[4]
                    
                    # Pitch hatası ve PID kontrolü
                    pitch_error = target_pitch - current_pitch
                    elevator_pwm = pitch_error * 0.01 + ver_vel * 0.02
                    
                    stabilizer.submit(self._elevator_control, elevator_pwm)

    def roll_stabilizer_function(self):
        """Roll stabilizasyon fonksiyonu"""
        with ThreadPoolExecutor(max_workers=self.num_roll_worker, 
                                thread_name_prefix="Roll") as stabilizer:
            while self.simulation_running:
                self.flight_data_event.wait()
                if self.stabilize_roll:
                    data = self.flight_data_stack[-1][0]
                    
                    # Hedef roll açısını belirle
                    target_roll = 0
                   # roll_stabilizer_function içinde
                    if self.navigation_active:
                        current_wp = self.waypoint_navigator.get_current_waypoint()
                        if current_wp:
                            current_lat = data[1]
                            current_lon = data[2]
                            current_yaw = data[9]
                            target_roll, _ = self.navigation_controller.get_navigation_commands(
                                current_lat, current_lon, current_yaw,
                                current_wp.latitude, current_wp.longitude)
                    else:
                        target_roll = 0
                    
                    # Roll kontrolü
                    roll = data[5]
                    roll_rate = data[6]
                    aileron_pwm = self.pid_aileron.get_aileron_pwm(roll, roll_rate, target_roll)
                    
                    stabilizer.submit(self._aileron_control, aileron_pwm)

    def set_stabilize_pitch(self, value: bool):
        """Pitch stabilizasyonunu aç/kapat"""
        if not isinstance(value, bool):
            raise TypeError("Boolean değer bekleniyor")
        self.stabilize_pitch = value

    def set_stabilize_roll(self, value: bool):
        """Roll stabilizasyonunu aç/kapat"""
        if not isinstance(value, bool):
            raise TypeError("Boolean değer bekleniyor")
        self.stabilize_roll = value

    def pitch_stabilization(self):
        """Pitch stabilizasyonu için thread fonksiyonu"""
        while True:
            if self.flight_data_stack:
                self.flight_data_event.wait()
                self.flight_data_event.clear()
                
                # Pitch stabilizasyonu
                self.pid_elevator.get_elevator_pwm(self.pitch, self.pitch_rate, 0)
                
                time.sleep(0.1)  # Her döngüde 100ms bekleme ekledim
            else:
                time.sleep(0.1)  # Veri yoksa 100ms bekle

    def roll_stabilization(self):
        """Roll stabilizasyonu için thread fonksiyonu"""
        while True:
            if self.flight_data_stack:
                self.flight_data_event.wait()
                self.flight_data_event.clear()
                
                # Roll stabilizasyonu
                self.pid_aileron.get_aileron_pwm(self.roll, self.roll_rate, 0)
                
                time.sleep(0.1)  # Her döngüde 100ms bekleme ekledim
            else:
                time.sleep(0.1)  # Veri yoksa 100ms bekle


# KULLANIM ÖRNEĞİ
if __name__ == "__main__":
    # TB2 İHA'yı başlat
    drone = TB2(display_data=True)
    
    # Başlangıç konumu: Latitude: -5.454957, Longitude: -146.145827
    # Hedef konumu: Latitude: -5.229957, Longitude: -145.920827
    
    # 1. Waypoint - Kuzeydoğuya 6km
    drone.add_waypoint(-5.409957, -146.100827, tolerance=300)
    
    # 2. Waypoint - Kuzeydoğuya 12km
    drone.add_waypoint(-5.364957, -146.055827, tolerance=300)
    
    # 3. Waypoint - Kuzeydoğuya 18km
    drone.add_waypoint(-5.319957, -146.010827, tolerance=300)
    
    # 4. Waypoint - Final hedef noktası
    drone.add_waypoint(-5.229957, -145.920827, tolerance=5000)
    
    print("Waypoint rotası hazırlandı:")
    print("Başlangıç: -5.454°, -146.145°")
    print("1. Hedef: Kuzeydoğu (6km)")
    print("2. Hedef: Kuzeydoğu (12km)")
    print("3. Hedef: Kuzeydoğu (18km)")
    print("4. Hedef: Final hedef noktası")
    
    # Navigasyonu başlat
    drone.start_navigation()
    
    # İHA'yı çalıştır
    drone.run()