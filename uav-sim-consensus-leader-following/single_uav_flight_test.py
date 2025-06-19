import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from enum import Enum
import random
import math

SIM_WIDTH = 350.0
SIM_HEIGHT = 350.0
TIME_STEP = 0.1
MAX_SIM_TIME = 80.0

BASE_CENTER_XY = np.array([50.0, SIM_HEIGHT / 2])
BASE_ALTITUDE = 0.0

RUNWAY_START_XY = np.array([BASE_CENTER_XY[0] - 50.0, SIM_HEIGHT / 2])
RUNWAY_END_XY = np.array([BASE_CENTER_XY[0] + 50.0, SIM_HEIGHT / 2])
RUNWAY_WIDTH = 10.0
RUNWAY_HEADING = math.atan2(RUNWAY_END_XY[1] - RUNWAY_START_XY[1], RUNWAY_END_XY[0] - RUNWAY_START_XY[0])

NUM_UAVS = 1
LEADER_ID = 0
UAV_MAX_SPEED_XY = 22.0
UAV_MIN_SPEED_XY = 8.0
UAV_ACCELERATION_XY = 2.5
UAV_DECELERATION_XY = 2.5
UAV_MAX_VERTICAL_SPEED = 3.5
UAV_MAX_TURN_RATE = math.radians(18.0)
UAV_TAKEOFF_SPEED = 16.0

UAV_RADIUS_VIS = 2.0
UAV_CRUISE_ALTITUDE = 60.0
UAV_POSITIONING_TOLERANCE_XY = 4.0
UAV_ALTITUDE_TOLERANCE = 2.0
UAV_HEADING_TOLERANCE = math.radians(15.0)
UAV_PATH_HISTORY_LENGTH = 250

KP_HEADING = 0.7
KP_ALT_FLIGHT = 0.9

class UAVStatus(Enum):
    IDLE_ON_GROUND = 0
    ACCELERATING_ON_RUNWAY = 1
    CLIMBING_AFTER_TAKEOFF = 2
    CRUISING = 3
    WAYPOINT_REACHED_TRANSITION = 4

def normalize_angle(angle):
    while angle > math.pi: angle -= 2 * math.pi
    while angle < -math.pi: angle += 2 * math.pi
    return angle

def angle_diff(angle1, angle2):
    return normalize_angle(angle1 - angle2)

class UAV:
    def __init__(self, id, initial_pos_xy, initial_alt, initial_heading):
        self.id = id
        self.position_xy = np.array(initial_pos_xy, dtype=float)
        self.altitude = float(initial_alt)
        self.heading = normalize_angle(float(initial_heading))
        self.current_speed_xy = 0.0
        self.status = UAVStatus.IDLE_ON_GROUND
        self.current_waypoint_xy = None
        self.current_target_altitude = initial_alt
        self.target_speed_xy = 0.0
        self.vertical_speed = 0.0
        self.path_history = []

    def _add_to_path_history(self):
        self.path_history.append(self.position_xy.copy())
        if len(self.path_history) > UAV_PATH_HISTORY_LENGTH:
            self.path_history.pop(0)

    def _update_flight_dynamics(self, desired_heading, desired_speed, desired_altitude, time_step_arg):
        heading_error = angle_diff(desired_heading, self.heading)
        effective_max_turn_rate = UAV_MAX_TURN_RATE

        if self.current_speed_xy < UAV_MIN_SPEED_XY and self.status != UAVStatus.ACCELERATING_ON_RUNWAY:
            speed_ratio = self.current_speed_xy / UAV_MIN_SPEED_XY
            effective_max_turn_rate *= max(0.2, speed_ratio ** 1.5)

        turn_amount = np.clip(KP_HEADING * heading_error, -effective_max_turn_rate, effective_max_turn_rate)
        self.heading = normalize_angle(self.heading + turn_amount * time_step_arg)

        if self.status == UAVStatus.ACCELERATING_ON_RUNWAY:
            self.current_speed_xy += UAV_ACCELERATION_XY * time_step_arg
        else:
            speed_error = desired_speed - self.current_speed_xy
            if speed_error > UAV_ACCELERATION_XY * time_step_arg * 0.1:
                self.current_speed_xy += UAV_ACCELERATION_XY * time_step_arg
            elif speed_error < -UAV_DECELERATION_XY * time_step_arg * 0.1:
                self.current_speed_xy -= UAV_DECELERATION_XY * time_step_arg
            elif abs(speed_error) < UAV_ACCELERATION_XY * time_step_arg:
                self.current_speed_xy = desired_speed

        if self.status.value > UAVStatus.ACCELERATING_ON_RUNWAY.value:
            self.current_speed_xy = np.clip(self.current_speed_xy, UAV_MIN_SPEED_XY, UAV_MAX_SPEED_XY)
        elif self.status == UAVStatus.ACCELERATING_ON_RUNWAY:
            self.current_speed_xy = np.clip(self.current_speed_xy, 0, UAV_MAX_SPEED_XY)
        else:
            self.current_speed_xy = 0.0

        alt_error = desired_altitude - self.altitude
        can_change_alt = (
                                     self.current_speed_xy >= UAV_TAKEOFF_SPEED * 0.80 and self.status.value >= UAVStatus.CLIMBING_AFTER_TAKEOFF.value) or \
                         (self.status == UAVStatus.CLIMBING_AFTER_TAKEOFF)

        if can_change_alt and abs(alt_error) > UAV_ALTITUDE_TOLERANCE:
            self.vertical_speed = np.clip(KP_ALT_FLIGHT * alt_error, -UAV_MAX_VERTICAL_SPEED, UAV_MAX_VERTICAL_SPEED)
        elif not can_change_alt and self.status.value >= UAVStatus.CLIMBING_AFTER_TAKEOFF.value and self.current_speed_xy < UAV_MIN_SPEED_XY:
            self.vertical_speed = np.clip(-UAV_MAX_VERTICAL_SPEED * 0.2, -UAV_MAX_VERTICAL_SPEED,
                                          UAV_MAX_VERTICAL_SPEED)
        else:
            self.vertical_speed = 0.0

        if self.status.value <= UAVStatus.ACCELERATING_ON_RUNWAY.value:
            self.altitude = BASE_ALTITUDE
            self.vertical_speed = 0.0
        else:
            self.altitude += self.vertical_speed * time_step_arg
            self.altitude = max(BASE_ALTITUDE, self.altitude)

        self.velocity_xy = np.array([math.cos(self.heading), math.sin(self.heading)]) * self.current_speed_xy
        self.position_xy += self.velocity_xy * time_step_arg
        self._add_to_path_history()

    def update(self, time_step_arg):
        d_hdg = self.heading
        d_spd = self.target_speed_xy
        d_alt = self.current_target_altitude

        if self.current_waypoint_xy is not None and self.status.value >= UAVStatus.CLIMBING_AFTER_TAKEOFF.value:
            vec_to_wp = self.current_waypoint_xy - self.position_xy
            if np.linalg.norm(
                    vec_to_wp) > UAV_POSITIONING_TOLERANCE_XY * 0.1:
                d_hdg = math.atan2(vec_to_wp[1], vec_to_wp[0])

        if self.status == UAVStatus.ACCELERATING_ON_RUNWAY:
            d_hdg = RUNWAY_HEADING
            d_spd = UAV_MAX_SPEED_XY
            d_alt = BASE_ALTITUDE
        elif self.status == UAVStatus.CLIMBING_AFTER_TAKEOFF:
            d_spd = UAV_MAX_SPEED_XY * 0.85
            d_alt = self.current_target_altitude
        elif self.status == UAVStatus.CRUISING:
            d_spd = UAV_MAX_SPEED_XY * 0.8
            d_alt = self.current_target_altitude
        elif self.status == UAVStatus.IDLE_ON_GROUND:
            d_spd = 0.0
            d_alt = BASE_ALTITUDE

        self._update_flight_dynamics(d_hdg, d_spd, d_alt, time_step_arg)

uavs = []
simulation_time = 0.0

climb_out_distance = 80.0
wp1_distance_after_climbout = 120.0
wp2_offset_distance = 100.0
wp2_turn_angle_from_wp1_heading = math.radians(60)

test_waypoint_climbout = None
test_waypoint_1 = None
test_waypoint_2 = None

def initialize_simulation():
    global uavs, simulation_time, test_waypoint_climbout, test_waypoint_1, test_waypoint_2
    uavs = []
    uav = UAV(LEADER_ID, RUNWAY_START_XY.copy(), BASE_ALTITUDE, initial_heading=RUNWAY_HEADING)
    uav.status = UAVStatus.IDLE_ON_GROUND
    uavs.append(uav)

    runway_vector = RUNWAY_END_XY - RUNWAY_START_XY
    runway_dir_norm = runway_vector / np.linalg.norm(runway_vector) if np.linalg.norm(runway_vector) > 0 else np.array(
        [1.0, 0.0])

    test_waypoint_climbout = RUNWAY_END_XY + runway_dir_norm * climb_out_distance
    test_waypoint_1 = test_waypoint_climbout + runway_dir_norm * wp1_distance_after_climbout

    vec_co_to_wp1 = test_waypoint_1 - test_waypoint_climbout
    heading_co_to_wp1 = math.atan2(vec_co_to_wp1[1], vec_co_to_wp1[0])

    wp2_heading = normalize_angle(heading_co_to_wp1 + wp2_turn_angle_from_wp1_heading)
    wp2_dir = np.array([math.cos(wp2_heading), math.sin(wp2_heading)])
    test_waypoint_2 = test_waypoint_1 + wp2_dir * wp2_offset_distance

    simulation_time = 0.0
    print("Sim Initialized: Single UAV Takeoff & Flight Test (Rev 2)")
    print(f"Runway Hdg: {math.degrees(RUNWAY_HEADING):.1f}°, UAV Status: {uavs[0].status.name}")
    print(f"WP ClimbOut: {test_waypoint_climbout}, WP1: {test_waypoint_1}, WP2: {test_waypoint_2}")

initialize_simulation()

fig, ax = plt.subplots(figsize=(12, 10))
uav_artists = []
uav_altitude_texts = []
uav_path_lines = []
uav_direction_lines = []

for i in range(NUM_UAVS):
    color = 'red'
    artist, = ax.plot([], [], 'o', markersize=10, color=color, label=f"UAV {i} (L)")
    uav_artists.append(artist)
    alt_text = ax.text(0, 0, "", fontsize=8, ha='center', va='bottom')
    uav_altitude_texts.append(alt_text)
    path_line, = ax.plot([], [], '--', linewidth=1.0, color=color, alpha=0.7)
    uav_path_lines.append(path_line)
    dir_line, = ax.plot([], [], '-', linewidth=1.5, color=color, alpha=0.9)
    uav_direction_lines.append(dir_line)

runway_patch = patches.Rectangle(RUNWAY_START_XY - np.array([0, RUNWAY_WIDTH / 2]),
                                 np.linalg.norm(RUNWAY_END_XY - RUNWAY_START_XY), RUNWAY_WIDTH,
                                 angle=math.degrees(RUNWAY_HEADING), color='gray', alpha=0.5, label="Runway")
ax.add_patch(runway_patch)
wp_climbout_artist, = ax.plot([], [], 'ms', markersize=7, label="WP ClimbOut")
wp1_artist, = ax.plot([], [], 'gs', markersize=7, label="Waypoint 1")
wp2_artist, = ax.plot([], [], 'bs', markersize=7, label="Waypoint 2")

ax.set_xlim(0, SIM_WIDTH)
ax.set_ylim(0, SIM_HEIGHT)
ax.set_aspect('equal', adjustable='box')
ax.legend(fontsize=8, loc='upper right')
title_text = ax.set_title(f"Time: {0.0:.1f}s | Status: IDLE_ON_GROUND")
plt.tight_layout()

def update_simulation(frame_num):
    global simulation_time
    simulation_time += TIME_STEP
    uav = uavs[0]

    current_wp_for_uav = uav.current_waypoint_xy

    if uav.status == UAVStatus.IDLE_ON_GROUND:
        if simulation_time > 0.5:
            uav.status = UAVStatus.ACCELERATING_ON_RUNWAY
            uav.target_speed_xy = UAV_MAX_SPEED_XY
            uav.current_target_altitude = BASE_ALTITUDE
            print(f"T+{simulation_time:.1f}s: UAV {uav.id} ACCEL_RUNWAY, TgtSpd: {uav.target_speed_xy}")

    elif uav.status == UAVStatus.ACCELERATING_ON_RUNWAY:
        uav.current_target_altitude = BASE_ALTITUDE
        uav.target_speed_xy = UAV_MAX_SPEED_XY
        if uav.current_speed_xy >= UAV_TAKEOFF_SPEED:
            uav.status = UAVStatus.CLIMBING_AFTER_TAKEOFF
            uav.current_target_altitude = UAV_CRUISE_ALTITUDE
            uav.target_speed_xy = UAV_MAX_SPEED_XY * 0.9
            uav.current_waypoint_xy = test_waypoint_climbout
            print(
                f"T+{simulation_time:.1f}s: UAV {uav.id} CLIMBING, TgtAlt: {uav.current_target_altitude}, TgtWP: {uav.current_waypoint_xy}")
        elif np.linalg.norm(uav.position_xy - RUNWAY_END_XY) < 5.0 and uav.current_speed_xy < UAV_TAKEOFF_SPEED:
            print(f"T+{simulation_time:.1f}s: UAV {uav.id} Ran out of runway!")
            uav.status = UAVStatus.IDLE_ON_GROUND
            uav.target_speed_xy = 0.0

    elif uav.status == UAVStatus.CLIMBING_AFTER_TAKEOFF:
        uav.current_target_altitude = UAV_CRUISE_ALTITUDE
        uav.target_speed_xy = UAV_MAX_SPEED_XY * 0.9
        uav.current_waypoint_xy = test_waypoint_climbout
        if abs(uav.altitude - UAV_CRUISE_ALTITUDE) < UAV_ALTITUDE_TOLERANCE and \
                (current_wp_for_uav is None or np.linalg.norm(
                    uav.position_xy - current_wp_for_uav) < UAV_POSITIONING_TOLERANCE_XY):
            uav.status = UAVStatus.CRUISING
            uav.current_waypoint_xy = test_waypoint_1
            uav.target_speed_xy = UAV_MAX_SPEED_XY * 0.8
            print(f"T+{simulation_time:.1f}s: UAV {uav.id} CRUISING to WP1: {uav.current_waypoint_xy}")

    elif uav.status == UAVStatus.CRUISING:
        uav.current_target_altitude = UAV_CRUISE_ALTITUDE
        uav.target_speed_xy = UAV_MAX_SPEED_XY * 0.8

        if current_wp_for_uav is None:
            uav.target_speed_xy = UAV_MIN_SPEED_XY
        elif np.linalg.norm(uav.position_xy - current_wp_for_uav) < UAV_POSITIONING_TOLERANCE_XY:
            uav.status = UAVStatus.WAYPOINT_REACHED_TRANSITION
            print(f"T+{simulation_time:.1f}s: UAV {uav.id} Reached WP: {current_wp_for_uav}. Transitioning.")

    elif uav.status == UAVStatus.WAYPOINT_REACHED_TRANSITION:
        if np.array_equal(current_wp_for_uav, test_waypoint_1):
            uav.current_waypoint_xy = test_waypoint_2
            uav.status = UAVStatus.CRUISING
            print(f"T+{simulation_time:.1f}s: UAV {uav.id} Now targeting WP2: {uav.current_waypoint_xy}")
        elif np.array_equal(current_wp_for_uav, test_waypoint_2):
            print(f"T+{simulation_time:.1f}s: UAV {uav.id} Reached Final WP2. Test End.")
            uav.current_waypoint_xy = None
            uav.target_speed_xy = UAV_MIN_SPEED_XY
            uav.status = UAVStatus.CRUISING
            if anim.event_source: anim.event_source.stop()
        else:
            uav.current_waypoint_xy = test_waypoint_1
            uav.status = UAVStatus.CRUISING
            print(
                f"T+{simulation_time:.1f}s: UAV {uav.id} (from transition) Now targeting WP1: {uav.current_waypoint_xy}")

    uav.update(TIME_STEP)

    uav_artists[0].set_data([uav.position_xy[0]], [uav.position_xy[1]])
    uav_altitude_texts[0].set_position((uav.position_xy[0], uav.position_xy[1] + UAV_RADIUS_VIS * 2.5))
    status_name = uav.status.name.replace('_', ' ')
    uav_altitude_texts[0].set_text(
        f"Alt:{uav.altitude:.0f}m H:{math.degrees(uav.heading):.0f} V:{uav.current_speed_xy:.1f}\n{status_name[:12]}")

    if uav.path_history:
        path_x, path_y = zip(*uav.path_history)
        uav_path_lines[0].set_data(path_x, path_y)

    line_len = uav.current_speed_xy * 0.2 + 5
    dir_x = math.cos(uav.heading) * line_len
    dir_y = math.sin(uav.heading) * line_len
    uav_direction_lines[0].set_data([uav.position_xy[0], uav.position_xy[0] + dir_x],
                                    [uav.position_xy[1], uav.position_xy[1] + dir_y])

    if test_waypoint_climbout is not None: wp_climbout_artist.set_data([test_waypoint_climbout[0]],
                                                                       [test_waypoint_climbout[1]])
    if test_waypoint_1 is not None: wp1_artist.set_data([test_waypoint_1[0]], [test_waypoint_1[1]])
    if test_waypoint_2 is not None: wp2_artist.set_data([test_waypoint_2[0]], [test_waypoint_2[1]])

    title_text.set_text(f"Time: {simulation_time:.1f}s | UAV Status: {uav.status.name}")

    if simulation_time >= MAX_SIM_TIME:
        print(f"Max simulation time reached.")
        if anim.event_source: anim.event_source.stop()

    return uav_artists + uav_altitude_texts + uav_path_lines + uav_direction_lines + \
           [title_text, wp_climbout_artist, wp1_artist, wp2_artist, runway_patch]

anim = animation.FuncAnimation(fig, update_simulation, frames=int(MAX_SIM_TIME / TIME_STEP),
                               interval=max(1, int(TIME_STEP * 1000)), blit=False, repeat=False)
plt.tight_layout()
plt.show()