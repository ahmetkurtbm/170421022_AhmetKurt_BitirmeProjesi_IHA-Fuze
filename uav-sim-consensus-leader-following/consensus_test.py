import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from enum import Enum
import random
import math

SIM_WIDTH = 500.0
SIM_HEIGHT = 500.0
TIME_STEP = 0.1
MAX_SIM_TIME = 120.0

NUM_UAVS = 5
LEADER_ID = 0
UAV_MAX_SPEED = 25.0
UAV_MIN_AIRBORNE_SPEED = 10.0
UAV_ACCELERATION = 3.0
UAV_DECELERATION = 3.0
UAV_CLIMB_RATE = 3.0
UAV_DESCENT_RATE = 3.0
UAV_MAX_TURN_RATE = math.radians(20.0)

UAV_RADIUS_VIS = 2.0
UAV_INITIAL_ALTITUDE = 70.0

WP_PROXIMITY_XY = 6.0
WP_PROXIMITY_ALT = 3.0
HEADING_ALIGNMENT_TOLERANCE = math.radians(10.0)

KP_HEADING_CONTROL = 0.9
KP_SPEED_CONTROL = 0.5
KP_ALTITUDE_CONTROL = 1.0

UAV_PATH_HISTORY_LENGTH = 200

FORMATION_OFFSETS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([-15.0, -15.0, -2.0]),
    np.array([-15.0, 15.0, -2.0]),
    np.array([-30.0, -30.0, -4.0]),
    np.array([-30.0, 30.0, -4.0]),
]

BASE_ALTITUDE = 0.0

class UAVStatus(Enum):
    FLYING_TO_WAYPOINT = "Flying to WP"
    HOLDING_PATTERN = "Holding"

def normalize_angle(angle_rad):
    while angle_rad > math.pi: angle_rad -= 2 * math.pi
    while angle_rad < -math.pi: angle_rad += 2 * math.pi
    return angle_rad

def angle_diff_rad(angle1_rad, angle2_rad):
    return normalize_angle(angle1_rad - angle2_rad)

class UAV:
    def __init__(self, id, pos_xy, alt, hdg, is_leader=False, formation_offset_body=np.zeros(3)):
        self.id = id
        self.pos_xy = np.array(pos_xy, dtype=float)
        self.alt = float(alt)
        self.hdg_rad = normalize_angle(float(hdg))
        self.speed_xy = UAV_MIN_AIRBORNE_SPEED
        self.vertical_speed = 0.0
        self.is_leader = is_leader
        self.formation_offset_body = np.array(formation_offset_body, dtype=float)
        self.leader_ref = None
        self.status = UAVStatus.FLYING_TO_WAYPOINT
        self.current_waypoint_3d = None
        self.commanded_target_speed_xy = UAV_MIN_AIRBORNE_SPEED
        self.path_history_xy = []

    def _add_to_path_history(self):
        self.path_history_xy.append(self.pos_xy.copy())
        if len(self.path_history_xy) > UAV_PATH_HISTORY_LENGTH:
            self.path_history_xy.pop(0)

    def _control_heading(self, desired_heading_rad, time_step):
        heading_error_rad = angle_diff_rad(desired_heading_rad, self.hdg_rad)
        effective_max_turn_rate = UAV_MAX_TURN_RATE
        if self.speed_xy < UAV_MIN_AIRBORNE_SPEED * 1.2:
            effective_max_turn_rate *= max(0.2, self.speed_xy / (UAV_MIN_AIRBORNE_SPEED * 1.2))
        turn_this_step_rad = np.clip(KP_HEADING_CONTROL * heading_error_rad,
                                     -effective_max_turn_rate * time_step,
                                     effective_max_turn_rate * time_step)
        self.hdg_rad = normalize_angle(self.hdg_rad + turn_this_step_rad)

    def _control_speed(self, commanded_speed_xy_arg, time_step):
        speed_error = commanded_speed_xy_arg - self.speed_xy
        accel_threshold = UAV_ACCELERATION * time_step * 0.1
        decel_threshold = UAV_DECELERATION * time_step * 0.1
        if speed_error > accel_threshold:
            self.speed_xy += UAV_ACCELERATION * time_step
        elif speed_error < -decel_threshold:
            self.speed_xy -= UAV_DECELERATION * time_step
        else:
            self.speed_xy = commanded_speed_xy_arg
        self.speed_xy = np.clip(self.speed_xy, UAV_MIN_AIRBORNE_SPEED, UAV_MAX_SPEED)

    def _control_altitude(self, desired_altitude, time_step):
        alt_error = desired_altitude - self.alt
        if abs(alt_error) > WP_PROXIMITY_ALT * 0.5:
            if alt_error > 0:
                self.vertical_speed = min(UAV_CLIMB_RATE,
                                          KP_ALTITUDE_CONTROL * alt_error / time_step if time_step > 0 else UAV_CLIMB_RATE)
            else:
                self.vertical_speed = -min(UAV_DESCENT_RATE, KP_ALTITUDE_CONTROL * abs(
                    alt_error) / time_step if time_step > 0 else UAV_DESCENT_RATE)
        else:
            self.vertical_speed = 0
            self.alt = desired_altitude
        self.alt += self.vertical_speed * time_step
        self.alt = max(BASE_ALTITUDE, self.alt)

    def set_leader_reference(self, leader_uav_obj):
        if not self.is_leader:
            self.leader_ref = leader_uav_obj

    def _calculate_follower_target_state(self):
        if self.is_leader or not self.leader_ref:
            return self.current_waypoint_3d[:2] if self.current_waypoint_3d is not None else self.pos_xy, \
                   self.hdg_rad, self.commanded_target_speed_xy, \
                   self.current_waypoint_3d[2] if self.current_waypoint_3d is not None else self.alt

        leader = self.leader_ref
        offset_along = self.formation_offset_body[0]
        offset_across = self.formation_offset_body[1]
        leader_fwd_vec = np.array([math.cos(leader.hdg_rad), math.sin(leader.hdg_rad)])
        leader_left_vec = np.array([-math.sin(leader.hdg_rad), math.cos(leader.hdg_rad)])
        desired_pos_xy_world = leader.pos_xy + offset_along * leader_fwd_vec + offset_across * leader_left_vec
        desired_alt_world = leader.alt + self.formation_offset_body[2]
        desired_hdg_world = leader.hdg_rad
        vec_to_desired_slot = desired_pos_xy_world - self.pos_xy
        dist_to_desired_slot = np.linalg.norm(vec_to_desired_slot)

        if dist_to_desired_slot > WP_PROXIMITY_XY * 1.5:
            hdg_to_slot = math.atan2(vec_to_desired_slot[1], vec_to_desired_slot[0])
            if dist_to_desired_slot > WP_PROXIMITY_XY * 3:
                desired_hdg_world = hdg_to_slot
            else:
                angle_error_to_slot_heading = angle_diff_rad(hdg_to_slot, self.hdg_rad)
                if abs(angle_error_to_slot_heading) > math.radians(30):
                    desired_hdg_world = hdg_to_slot

        desired_spd_world = leader.commanded_target_speed_xy
        actual_offset_vec = self.pos_xy - leader.pos_xy
        projected_along_dist = np.dot(actual_offset_vec, leader_fwd_vec)
        desired_along_dist = offset_along
        along_error = desired_along_dist - projected_along_dist
        speed_correction = KP_SPEED_CONTROL * along_error
        desired_spd_world += speed_correction
        desired_spd_world = np.clip(desired_spd_world, UAV_MIN_AIRBORNE_SPEED, UAV_MAX_SPEED)
        return desired_pos_xy_world, desired_hdg_world, desired_spd_world, desired_alt_world

    def update_state(self, time_step):
        d_hdg_rad = self.hdg_rad
        d_spd_xy = self.commanded_target_speed_xy
        d_alt = self.alt
        target_wp_xy_for_heading = None

        if self.is_leader:
            if self.current_waypoint_3d is not None:
                target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                d_alt = self.current_waypoint_3d[2]
            else:
                d_spd_xy = UAV_MIN_AIRBORNE_SPEED
        else:
            if self.leader_ref:
                des_pos_xy, des_hdg, des_spd, des_alt = self._calculate_follower_target_state()
                self.current_waypoint_3d = np.array([des_pos_xy[0], des_pos_xy[1], des_alt])
                target_wp_xy_for_heading = des_pos_xy
                d_hdg_rad = des_hdg
                d_spd_xy = des_spd
                d_alt = des_alt
            else:
                d_spd_xy = UAV_MIN_AIRBORNE_SPEED

        if target_wp_xy_for_heading is not None:
            vec_to_wp_xy = target_wp_xy_for_heading - self.pos_xy
            if np.linalg.norm(vec_to_wp_xy) > WP_PROXIMITY_XY * 0.2:
                d_hdg_rad = math.atan2(vec_to_wp_xy[1], vec_to_wp_xy[0])

        self.commanded_target_speed_xy = d_spd_xy
        self._control_heading(d_hdg_rad, time_step)
        self._control_speed(self.commanded_target_speed_xy, time_step)
        self._control_altitude(d_alt, time_step)

        self.pos_xy += np.array([math.cos(self.hdg_rad), math.sin(self.hdg_rad)]) * self.speed_xy * time_step
        self._add_to_path_history()

uavs = []
simulation_time = 0.0
leader_waypoints_3d = []

def initialize_simulation():
    global uavs, simulation_time, leader_waypoints_3d
    uavs = []
    leader = None

    leader_waypoints_3d = [
        np.array([50.0, SIM_HEIGHT / 2, UAV_INITIAL_ALTITUDE]),
        np.array([SIM_WIDTH * 0.4, SIM_HEIGHT * 0.6, UAV_INITIAL_ALTITUDE + 10]),
        np.array([SIM_WIDTH * 0.7, SIM_HEIGHT * 0.5, UAV_INITIAL_ALTITUDE + 10]),
        np.array([SIM_WIDTH * 0.8, SIM_HEIGHT * 0.3, UAV_INITIAL_ALTITUDE]),
        np.array([SIM_WIDTH * 0.5, SIM_HEIGHT * 0.2, UAV_INITIAL_ALTITUDE - 10]),
        np.array([SIM_WIDTH * 0.2, SIM_HEIGHT * 0.4, UAV_INITIAL_ALTITUDE - 5]),
    ]
    initial_heading_rad = math.atan2(leader_waypoints_3d[1][1] - leader_waypoints_3d[0][1],
                                     leader_waypoints_3d[1][0] - leader_waypoints_3d[0][0])

    for i in range(NUM_UAVS):
        is_l = (i == LEADER_ID)
        leader_start_pos_xy = leader_waypoints_3d[0][:2]
        leader_start_alt = leader_waypoints_3d[0][2]
        leader_start_hdg = initial_heading_rad
        if is_l:
            start_pos_xy, start_alt, start_hdg = leader_start_pos_xy, leader_start_alt, leader_start_hdg
        else:
            offset_body = FORMATION_OFFSETS[i]
            offset_along, offset_across = offset_body[0], offset_body[1]
            rot_off_x = offset_along * math.cos(leader_start_hdg) - offset_across * math.sin(leader_start_hdg)
            rot_off_y = offset_along * math.sin(leader_start_hdg) + offset_across * math.cos(leader_start_hdg)
            start_pos_xy = leader_start_pos_xy + np.array([rot_off_x, rot_off_y])
            start_alt = leader_start_alt + offset_body[2]
            start_hdg = leader_start_hdg
        uav = UAV(i, start_pos_xy, start_alt, start_hdg, is_leader=is_l, formation_offset_body=FORMATION_OFFSETS[i])
        if is_l:
            leader = uav
            uav.current_waypoint_3d = leader_waypoints_3d[1]
            uav.commanded_target_speed_xy = UAV_MAX_SPEED * 0.8
        uavs.append(uav)

    for uav_obj in uavs:
        if not uav_obj.is_leader and leader:
            uav_obj.set_leader_reference(leader)
            uav_obj.commanded_target_speed_xy = UAV_MAX_SPEED * 0.8
    simulation_time = 0.0
    if leader:
        leader.current_waypoint_idx = 1
    print("Sim Initialized: Multi-UAV Leader-Follower Test (AttributeError Fixed)")

initialize_simulation()

fig, ax = plt.subplots(figsize=(12, 10))
uav_artists = []
uav_altitude_texts = []
uav_path_lines = []
uav_direction_lines = []
for i in range(NUM_UAVS):
    color = 'red' if uavs[i].is_leader else ('blue' if i % 2 == 1 else 'green')
    marker_char = 'P' if uavs[i].is_leader else 'o'
    artist, = ax.plot([], [], marker=marker_char, markersize=8 if uavs[i].is_leader else 7,
                      color=color, label=f"UAV {i}{' (L)' if uavs[i].is_leader else F' (F{i})'}")
    uav_artists.append(artist)
    alt_text = ax.text(0, 0, "", fontsize=7, ha='center', va='bottom')
    uav_altitude_texts.append(alt_text)
    path_line, = ax.plot([], [], '--', lw=0.7, color=color, alpha=0.5)
    uav_path_lines.append(path_line)
    dir_line, = ax.plot([], [], '-', lw=1.0, color=color, alpha=0.7)
    uav_direction_lines.append(dir_line)

leader_wp_artists = []
for i, wp in enumerate(leader_waypoints_3d):
    artist, = ax.plot([wp[0]], [wp[1]], 'x', markersize=7, color='black', alpha=0.6,
                      label=f"L_WP{i}" if i == 0 and leader_waypoints_3d else None)
    leader_wp_artists.append(artist)

ax.set_xlim(0, SIM_WIDTH)
ax.set_ylim(0, SIM_HEIGHT)
ax.set_aspect('equal')
ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.01))
title_text = ax.set_title("")
plt.subplots_adjust(right=0.82)

def update_simulation(frame_num):
    global simulation_time
    simulation_time += TIME_STEP
    leader = uavs[LEADER_ID]

    if leader.current_waypoint_3d is not None:
        dist_to_wp_xy = np.linalg.norm(leader.pos_xy - leader.current_waypoint_3d[:2])
        if dist_to_wp_xy < WP_PROXIMITY_XY * 1.5:
            leader.current_waypoint_idx += 1
            if leader.current_waypoint_idx < len(leader_waypoints_3d):
                leader.current_waypoint_3d = leader_waypoints_3d[leader.current_waypoint_idx]
                leader.commanded_target_speed_xy = UAV_MAX_SPEED * (0.7 + random.uniform(-0.1, 0.1))
            else:
                leader.current_waypoint_3d = None
                leader.commanded_target_speed_xy = UAV_MIN_AIRBORNE_SPEED
                leader.status = UAVStatus.HOLDING_PATTERN
                print(f"T+{simulation_time:.1f}s: Leader completed all waypoints, holding.")
    elif leader.status != UAVStatus.HOLDING_PATTERN:
        leader.commanded_target_speed_xy = UAV_MIN_AIRBORNE_SPEED
        leader.status = UAVStatus.HOLDING_PATTERN

    for uav in uavs:
        uav.update_state(TIME_STEP)

    for i, uav in enumerate(uavs):
        uav_artists[i].set_data([uav.pos_xy[0]], [uav.pos_xy[1]])
        uav_altitude_texts[i].set_position((uav.pos_xy[0], uav.pos_xy[1] + UAV_RADIUS_VIS * 2.0))
        status_name_short = uav.status.name.replace('_', ' ')[:13]
        uav_altitude_texts[i].set_text(
            f"{uav.id}: {uav.alt:.0f}m H{math.degrees(uav.hdg_rad):.0f}° V{uav.speed_xy:.1f}\n{status_name_short}")
        if uav.path_history_xy:
            px, py = zip(*uav.path_history_xy)
            uav_path_lines[i].set_data(px, py)

        line_len = uav.speed_xy * 0.2 + 3

        dx = math.cos(uav.hdg_rad) * line_len
        dy = math.sin(uav.hdg_rad) * line_len
        uav_direction_lines[i].set_data([uav.pos_xy[0], uav.pos_xy[0] + dx], [uav.pos_xy[1], uav.pos_xy[1] + dy])

    title_text.set_text(f"Time: {simulation_time:.1f}s | Leader Status: {leader.status.name}")
    if simulation_time >= MAX_SIM_TIME:
        print(f"Max sim time. Final Leader Status: {leader.status.name}")
        if anim.event_source: anim.event_source.stop()
    return uav_artists + uav_altitude_texts + uav_path_lines + uav_direction_lines + leader_wp_artists + [title_text]

anim = animation.FuncAnimation(fig, update_simulation, frames=int(MAX_SIM_TIME / TIME_STEP),
                               interval=max(1, int(TIME_STEP * 150)), blit=False, repeat=False)
plt.show()