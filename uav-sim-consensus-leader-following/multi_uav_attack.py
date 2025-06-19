import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from enum import Enum
import random
import math

# --- Simulation Parameters ---
SIM_WIDTH = 600.0  # Width of the simulation area in meters
SIM_HEIGHT = 600.0  # Height of the simulation area in meters
TIME_STEP = 0.1  # Simulation time step in seconds
MAX_SIM_TIME = 350.0  # Maximum simulation duration in seconds

# --- UAV Parameters ---
NUM_UAVS = 5  # Total number of UAVs in the simulation
LEADER_ID = 0  # Initial ID of the leader UAV
UAV_MAX_SPEED = 28.0  # Maximum horizontal speed of a UAV in m/s
UAV_MIN_AIRBORNE_SPEED = 12.0  # Minimum airborne speed (stall speed approximation) in m/s
UAV_ACCELERATION = 2.5  # UAV acceleration rate in m/s^2
UAV_DECELERATION = 2.5  # UAV deceleration rate in m/s^2
UAV_CLIMB_RATE = 4.0  # Maximum vertical climb rate in m/s
UAV_DESCENT_RATE = 4.0  # Maximum vertical descent rate in m/s
UAV_MAX_TURN_RATE = math.radians(15.0)  # Maximum turn rate in radians/s

UAV_RADIUS_VIS = 2.0  # Visual radius for UAV representation on plot
UAV_INITIAL_ALTITUDE = 150.0  # Starting altitude for all UAVs in meters

# Tolerances for waypoint proximity and heading alignment
WP_PROXIMITY_XY = 15.0  # Horizontal distance tolerance to consider waypoint reached
WP_PROXIMITY_ALT = 5.0  # Altitude tolerance to consider waypoint reached
HEADING_ALIGNMENT_TOLERANCE = math.radians(10.0)  # Angular tolerance for heading alignment

# Control Gains (Proportional gains for PID-like control)
KP_HEADING_CONTROL = 0.9  # Gain for heading correction
KP_SPEED_CONTROL = 0.6  # Gain for speed correction
KP_ALTITUDE_CONTROL = 1.0  # Gain for altitude correction

UAV_PATH_HISTORY_LENGTH = 300  # Number of past positions to store for path visualization

# Formation Offsets: [along_leader_axis, across_leader_axis, alt_diff]
# Defines the relative position of each UAV within the formation, relative to the leader's body frame
FORMATION_OFFSETS = [
    np.array([0.0, 0.0, 0.0]),  # Leader (UAV 0) has no offset
    np.array([-25.0, -20.0, -5.0]),  # UAV 1: 25m behind, 20m left, 5m below leader
    np.array([-25.0, 20.0, -5.0]),  # UAV 2: 25m behind, 20m right, 5m below leader
    np.array([-50.0, -40.0, -10.0]),  # UAV 3: 50m behind, 40m left, 10m below leader
    np.array([-50.0, 40.0, -10.0]),  # UAV 4: 50m behind, 40m right, 10m below leader
]
BASE_ALTITUDE = 0.0  # Ground level altitude

# --- Mission Specific Parameters ---
GBAD_TARGET_POS_3D = np.array([SIM_WIDTH * 0.7, SIM_HEIGHT * 0.7,
                               UAV_INITIAL_ALTITUDE - 20.0])  # 3D position of the Ground-Based Air Defense (GBAD) target
RETURN_BASE_POS_3D = np.array(
    [SIM_WIDTH * 0.1, SIM_HEIGHT * 0.1, UAV_INITIAL_ALTITUDE - 50.0])  # 3D position of the return base

# --- No-Fly Zone (NFZ) Related Global Parameters ---
GBAD_EXCLUSION_RADIUS = 60.0  # Radius around GBAD target that is considered an exclusion zone
GBAD_AVOID_ANTICIPATION_MARGIN = 20.0  # Additional buffer for UAVs to start avoiding GBAD zone
POLYGON_NFZ_AVOID_ANTICIPATION_MARGIN = 20.0  # Additional buffer for UAVs to start avoiding polygon NFZ
ATTACK_POINT_BUFFER = 5.0  # Buffer distance for UAVs to engage target from a safe distance

# Vertices defining an irregular polygonal No-Fly Zone
IRREGULAR_NFZ_VERTICES = np.array([
    [SIM_WIDTH * 0.15, SIM_HEIGHT * 0.25], [SIM_WIDTH * 0.45, SIM_HEIGHT * 0.3],
    [SIM_WIDTH * 0.40, SIM_HEIGHT * 0.45], [SIM_WIDTH * 0.10, SIM_HEIGHT * 0.4]
])
IRREGULAR_NFZ_CENTROID = np.mean(IRREGULAR_NFZ_VERTICES, axis=0)  # Centroid of the polygon for calculations
IRREGULAR_NFZ_EFFECTIVE_RADIUS = np.max(
    np.linalg.norm(IRREGULAR_NFZ_VERTICES - IRREGULAR_NFZ_CENTROID, axis=1))  # Max distance from centroid to a vertex

# Initial calculation for LEADER_ATTACK_RELEASE_POINT_3D for "Leader Direct Attack" strategy
# This point is set relative to the GBAD target, outside its exclusion radius.
_vec_from_GBAD_origin_to_initial_rp_suggestion = np.array(
    [(GBAD_EXCLUSION_RADIUS + 70.0), (GBAD_EXCLUSION_RADIUS + 70.0), -10.0])
if np.linalg.norm(_vec_from_GBAD_origin_to_initial_rp_suggestion[:2]) < 1e-5:
    _norm_vec_to_GBAD_initial_rp_calc = np.array([1.0, 0.0, 0.0])
else:
    _norm_vec_to_GBAD_initial_rp_calc = _vec_from_GBAD_origin_to_initial_rp_suggestion / (
            np.linalg.norm(_vec_from_GBAD_origin_to_initial_rp_suggestion) + 1e-6)

LEADER_ATTACK_RELEASE_POINT_3D = GBAD_TARGET_POS_3D - _norm_vec_to_GBAD_initial_rp_calc * (
        GBAD_EXCLUSION_RADIUS + 40.0)
LEADER_ATTACK_RELEASE_POINT_3D[2] = GBAD_TARGET_POS_3D[2] + 10.0  # Ensure reasonable altitude

ATTACK_RUN_SPEED = UAV_MAX_SPEED * 0.9  # Speed for attack runs
LEADER_EVASION_DURATION = 5.0  # Duration of leader's evasion maneuver after attack

# GBAD Counter-Attack Parameters
GBAD_vulnerable_event_time = -1.0  # Time when GBAD becomes vulnerable (e.g., after initial attack)
GBAD_lock_on_start_time = -1.0  # Time when GBAD starts locking onto a UAV
GBAD_target_uav_id = -1  # ID of the UAV GBAD is currently locked onto
GBAD_LOCK_ON_DELAY = 0.5  # Delay before GBAD locks on after detecting a threat
GBAD_LOCK_ON_DURATION = 4.0  # Duration GBAD stays locked on a target
FOLLOWER_ENGAGE_ACTION_TIME = 2.0  # Duration followers spend "engaging" the target
GBAD_LOCK_ON_RANGE = 120.0  # Maximum range for GBAD to detect and lock onto a UAV


# Enumeration for UAV statuses, describing its current mission phase or state.
class UAVStatus(Enum):
    IDLE = "Idle"
    FLYING_TO_WAYPOINT = "Flying to WP"
    HOLDING_PATTERN = "Holding Pattern"
    APPROACHING_TARGET = "Approaching Target"
    ENGAGING_TARGET = "Engaging Target"
    POST_ENGAGE_CLEARING = "Clearing GBAD Post-Engage"  # Maneuver after engaging GBAD
    EVADING = "Evading"  # Evasive maneuvers
    WAITING_FOR_ENGAGE_ORDER = "Waiting Engage Order"  # Follower state, waiting for leader's action
    RETURNING_TO_BASE = "Returning to Base"
    LANDED = "Landed"
    AVOIDING_NFZ = "Avoiding NFZ"  # Actively maneuvering to stay out of a No-Fly Zone
    INCAPACITATED = "Incapacitated"  # UAV is disabled/destroyed
    MISSILE_DROPPED_INFO = "MISSILE DROPPED!"  # Temporary display status


# --- Helper Functions ---
def normalize_angle(angle_rad):
    # Ensures an angle is within the range [-pi, pi]
    while angle_rad > math.pi: angle_rad -= 2 * math.pi
    while angle_rad < -math.pi: angle_rad += 2 * math.pi
    return angle_rad


def angle_diff_rad(angle1_rad, angle2_rad):
    # Calculates the shortest angular difference between two angles
    return normalize_angle(angle1_rad - angle2_rad)


def is_point_inside_polygon(point_xy, polygon_vertices):
    # Ray casting algorithm to check if a point is inside a polygon
    x, y = point_xy
    n = len(polygon_vertices)
    inside = False
    p1x, p1y = polygon_vertices[0]
    for i in range(n + 1):
        p2x, p2y = polygon_vertices[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y: xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters: inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def dist_point_to_segment(px, py, x1, y1, x2, y2):
    # Calculates the shortest distance from a point (px, py) to a line segment (x1, y1)-(x2, y2).
    line_mag_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if line_mag_sq == 0:  # Segment is a point
        return np.linalg.norm(np.array([px - x1, py - y1]))

    # Project point onto the line defined by the segment
    t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_mag_sq
    t = max(0, min(1, t))  # Clamp t to [0, 1] to stay within segment boundaries

    closest_x = x1 + t * (x2 - x1)
    closest_y = y1 + t * (y2 - y1)

    return np.linalg.norm(np.array([px - closest_x, py - closest_y]))


def line_intersects_circle(p1_xy, p2_xy, circle_center_xy, circle_radius):
    # Checks if a line segment (representing a path) intersects a circle (representing a circular NFZ).
    return dist_point_to_segment(p1_xy[0], p1_xy[1], circle_center_xy[0], circle_center_xy[1], p2_xy[0],
                                 p2_xy[1]) <= circle_radius


def line_intersects_polygon(p1_xy, p2_xy, polygon_vertices):
    # Checks if a line segment intersects a polygon. Also checks if endpoints are inside.
    # Checks if either endpoint is inside the polygon
    if is_point_inside_polygon(p1_xy, polygon_vertices) or is_point_inside_polygon(p2_xy, polygon_vertices):
        return True

    # Check for intersection with each edge of the polygon using cross-product/orientation test
    n = len(polygon_vertices)
    for i in range(n):
        p3_xy = polygon_vertices[i]
        p4_xy = polygon_vertices[(i + 1) % n]

        # Orientation test: Check if (p1, p2) and (p3, p4) intersect
        def ccw(A, B, C):  # Counter-clockwise test
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        if (ccw(p1_xy, p3_xy, p4_xy) != ccw(p2_xy, p3_xy, p4_xy)) and \
                (ccw(p1_xy, p2_xy, p3_xy) != ccw(p1_xy, p2_xy, p4_xy)):
            return True
    return False


# --- UAV Class Definition ---
class UAV:
    def __init__(self, id, pos_xy, alt, hdg, is_leader=False, formation_offset_body=np.zeros(3)):
        self.id = id  # Unique identifier for the UAV
        self.pos_xy = np.array(pos_xy, dtype=float)  # Current 2D position (x, y)
        self.alt = float(alt)  # Current altitude
        self.hdg_rad = normalize_angle(float(hdg))  # Current heading in radians
        self.speed_xy = UAV_MIN_AIRBORNE_SPEED  # Current horizontal speed
        self.vertical_speed = 0.0  # Current vertical speed
        self.is_leader = is_leader  # True if this UAV is the leader
        self.formation_offset_body = np.array(formation_offset_body,
                                              dtype=float)  # Desired offset from leader (body frame)
        self.leader_ref = None  # Reference to the leader UAV object (if a follower)
        self.status = UAVStatus.FLYING_TO_WAYPOINT  # Current mission status
        self.current_waypoint_3d = None  # Current target waypoint (x, y, alt)
        self.commanded_target_speed_xy = UAV_MIN_AIRBORNE_SPEED  # Desired horizontal speed for control
        self.path_history_xy = []  # List to store past positions for path visualization
        self.action_timer = 0.0  # Generic timer for timed actions (e.g., engage duration, evasion)
        self.previous_status_before_avoidance = None  # Stores status before entering NFZ avoidance
        self.evasion_heading_rad = 0.0  # Specific heading for evasion
        self.is_on_detour_to_base = False  # Flag for RTB detour around NFZ
        self.is_active = True  # False if UAV is incapacitated
        self.currently_avoiding_nfz_details = None  # Details of the NFZ currently being avoided
        self.must_rtb_after_nfz_violation = False  # Flag to force RTB after an NFZ violation (e.g., leader)
        self.missile_dropped_display_timer = 0.0  # Timer for displaying "MISSILE DROPPED!" text

    def _add_to_path_history(self):
        # Adds the current position to the path history for visualization.
        self.path_history_xy.append(self.pos_xy.copy())
        if len(self.path_history_xy) > UAV_PATH_HISTORY_LENGTH: self.path_history_xy.pop(0)

    def _control_heading(self, desired_heading_rad, time_step):
        # Controls the UAV's heading towards a desired angle.
        heading_error_rad = angle_diff_rad(desired_heading_rad, self.hdg_rad)

        # Dampen turn rate when nearing the base or at low speeds for smoother control.
        effective_max_turn_rate = UAV_MAX_TURN_RATE
        if self.status in [UAVStatus.RETURNING_TO_BASE, UAVStatus.LANDED]:
            dist_to_wp = np.linalg.norm(self.pos_xy - RETURN_BASE_POS_3D[:2])
            if dist_to_wp < WP_PROXIMITY_XY * 3:
                dampening_factor = max(0.1, dist_to_wp / (WP_PROXIMITY_XY * 3))
                effective_max_turn_rate *= dampening_factor

        turn_capability_factor = max(0.1, self.speed_xy / (UAV_MIN_AIRBORNE_SPEED * 1.5))
        effective_max_turn_rate *= turn_capability_factor  # Reduce turn rate if speed is too low

        # Apply proportional control and clip to max turn rate.
        turn_this_step_rad = KP_HEADING_CONTROL * heading_error_rad
        turn_this_step_rad = np.clip(turn_this_step_rad, -effective_max_turn_rate * time_step,
                                     effective_max_turn_rate * time_step)
        self.hdg_rad = normalize_angle(self.hdg_rad + turn_this_step_rad)

    def _control_speed(self, commanded_speed_xy_arg, time_step):
        # Controls the UAV's horizontal speed towards a commanded value.
        # Dampen speed when nearing the base for a smoother landing approach.
        if self.status in [UAVStatus.RETURNING_TO_BASE, UAVStatus.LANDED]:
            dist_to_wp = np.linalg.norm(self.pos_xy - RETURN_BASE_POS_3D[:2])
            if dist_to_wp < WP_PROXIMITY_XY * 2:
                commanded_speed_xy_arg = commanded_speed_xy_arg * (dist_to_wp / (WP_PROXIMITY_XY * 2))
                commanded_speed_xy_arg = max(UAV_MIN_AIRBORNE_SPEED * 0.2, commanded_speed_xy_arg)

        speed_error = commanded_speed_xy_arg - self.speed_xy
        # Apply acceleration/deceleration with small thresholds to prevent jittering.
        if speed_error > UAV_ACCELERATION * time_step * 0.1:
            self.speed_xy += UAV_ACCELERATION * time_step
        elif speed_error < -UAV_DECELERATION * time_step * 0.1:
            self.speed_xy -= UAV_DECELERATION * time_step
        else:
            self.speed_xy = commanded_speed_xy_arg
        # Ensure speed stays within min/max limits.
        self.speed_xy = np.clip(self.speed_xy, UAV_MIN_AIRBORNE_SPEED, UAV_MAX_SPEED)

    def _control_altitude(self, desired_altitude, time_step):
        # Controls the UAV's altitude towards a desired value.
        alt_error = desired_altitude - self.alt
        if abs(alt_error) > WP_PROXIMITY_ALT * 0.5:  # Only adjust if error is significant
            # Apply proportional control for vertical speed, clamped by max climb/descent rates.
            if alt_error > 0:
                self.vertical_speed = min(UAV_CLIMB_RATE,
                                          KP_ALTITUDE_CONTROL * alt_error / time_step if time_step > 0 else UAV_CLIMB_RATE)
            else:
                self.vertical_speed = -min(UAV_DESCENT_RATE, KP_ALTITUDE_CONTROL * abs(
                    alt_error) / time_step if time_step > 0 else UAV_DESCENT_RATE)
        else:
            self.vertical_speed = 0.0  # Stop vertical movement if close enough
            self.alt = desired_altitude  # Snap to desired altitude to prevent oscillation
        self.alt += self.vertical_speed * time_step  # Update altitude
        self.alt = max(BASE_ALTITUDE, self.alt)  # Prevent going below ground

    def set_leader_reference(self, leader_uav_obj):
        # Sets the reference to the leader UAV for followers.
        if not self.is_leader: self.leader_ref = leader_uav_obj

    def _calculate_follower_target_state(self):
        # Calculates the desired position, heading, speed, and altitude for a follower UAV.
        if self.is_leader or not self.leader_ref:  # If it's the leader or no leader reference, behave as autonomous
            return (self.current_waypoint_3d[:2] if self.current_waypoint_3d is not None else self.pos_xy,
                    self.hdg_rad, self.commanded_target_speed_xy,
                    self.current_waypoint_3d[2] if self.current_waypoint_3d is not None else self.alt)

        leader = self.leader_ref
        offset_along, offset_across = self.formation_offset_body[0], self.formation_offset_body[1]

        # Calculate leader's forward and left vectors in world coordinates
        leader_fwd_vec = np.array([math.cos(leader.hdg_rad), math.sin(leader.hdg_rad)])
        leader_left_vec = np.array([-math.sin(leader.hdg_rad), math.cos(leader.hdg_rad)])

        # Calculate desired world position based on leader's position and formation offset
        desired_pos_xy_world = leader.pos_xy + offset_along * leader_fwd_vec + offset_across * leader_left_vec
        desired_alt_world = leader.alt + self.formation_offset_body[2]
        desired_hdg_world = leader.hdg_rad  # Follower tries to match leader's heading

        # Adjust heading if far from desired slot to quickly catch up
        vec_to_desired_slot = desired_pos_xy_world - self.pos_xy
        dist_to_desired_slot = np.linalg.norm(vec_to_desired_slot)
        if dist_to_desired_slot > WP_PROXIMITY_XY * 1.5:
            hdg_to_slot = math.atan2(vec_to_desired_slot[1], vec_to_desired_slot[0])
            if dist_to_desired_slot > WP_PROXIMITY_XY * 3.0 or abs(
                    angle_diff_rad(hdg_to_slot, self.hdg_rad)) > math.radians(30):
                desired_hdg_world = hdg_to_slot

        # Adjust speed to maintain longitudinal position within formation
        desired_spd_world = leader.commanded_target_speed_xy
        projected_along_dist = np.dot(self.pos_xy - leader.pos_xy, leader_fwd_vec)
        speed_correction = KP_SPEED_CONTROL * (offset_along - projected_along_dist)
        desired_spd_world = np.clip(desired_spd_world + speed_correction, UAV_MIN_AIRBORNE_SPEED, UAV_MAX_SPEED)

        return desired_pos_xy_world, desired_hdg_world, desired_spd_world, desired_alt_world

    def update_state(self, time_step):
        # Main update function for each UAV, called every simulation step.
        # It manages UAV's state transitions, NFZ avoidance, and calls control functions.
        global GBAD_vulnerable_event_time, GBAD_lock_on_start_time, uavs, LEADER_ID, SIM_ATTACK_STRATEGY, NFZ_DEFINITIONS, GBAD_EXCLUSION_RADIUS, ATTACK_POINT_BUFFER, GBAD_target_uav_id

        if not self.is_active:  # If UAV is incapacitated, it stops moving.
            self.speed_xy = 0
            self.vertical_speed = 0
            if self.status != UAVStatus.INCAPACITATED: self.status = UAVStatus.INCAPACITATED
            return

        # Decrement missile drop display timer
        if self.missile_dropped_display_timer > 0:
            self.missile_dropped_display_timer -= time_step
            if self.missile_dropped_display_timer <= 0:
                pass  # Timer ended, nothing to do.

        # Initialize desired state variables.
        d_hdg_rad = self.hdg_rad
        d_spd_xy = self.commanded_target_speed_xy
        d_alt = self.alt
        target_wp_xy_for_heading = None

        # If a follower's leader is returning to base, the follower also initiates RTB.
        if not self.is_leader and self.leader_ref and self.leader_ref.is_active and \
                self.leader_ref.status == UAVStatus.RETURNING_TO_BASE and \
                self.status not in [UAVStatus.RETURNING_TO_BASE, UAVStatus.LANDED, UAVStatus.AVOIDING_NFZ]:
            self.status = UAVStatus.RETURNING_TO_BASE
            self.current_waypoint_3d = RETURN_BASE_POS_3D
            self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.75
            self.is_on_detour_to_base = False
            self.action_timer = 0

        # --- NFZ Avoidance Logic ---
        # Checks if UAV needs to avoid any No-Fly Zones.
        nfz_violation_detected_this_step = False
        if SIM_INCLUDE_NFZ and NFZ_DEFINITIONS and self.status not in [UAVStatus.AVOIDING_NFZ,
                                                                       UAVStatus.RETURNING_TO_BASE]:
            # Predict short-term future position.
            next_pos_estimate_short = self.pos_xy + np.array(
                [math.cos(self.hdg_rad), math.sin(self.hdg_rad)]) * self.speed_xy * TIME_STEP * 2.0

            for i_nfz, nfz_def in enumerate(NFZ_DEFINITIONS):
                # Special handling for GBAD Zone NFZ: allow legitimate attacks to proceed.
                if nfz_def["name"] == "GBAD Zone":
                    is_legit_GBAD_approach = False
                    if self.current_waypoint_3d is not None:
                        if SIM_ATTACK_STRATEGY == "Leader Direct Attack" and self.is_leader and \
                                (
                                        self.status == UAVStatus.APPROACHING_TARGET or self.status == UAVStatus.ENGAGING_TARGET) and \
                                np.linalg.norm(self.current_waypoint_3d[:2] - LEADER_ATTACK_RELEASE_POINT_3D[
                                                                              :2]) < WP_PROXIMITY_XY * 2:
                            is_legit_GBAD_approach = True
                        elif SIM_ATTACK_STRATEGY == "Leader Direct Attack" and not self.is_leader and self.status == UAVStatus.ENGAGING_TARGET and \
                                np.linalg.norm(self.pos_xy - GBAD_TARGET_POS_3D[:2]) < (
                                GBAD_EXCLUSION_RADIUS + ATTACK_POINT_BUFFER + WP_PROXIMITY_XY * 2):
                            is_legit_GBAD_approach = True
                        elif SIM_ATTACK_STRATEGY == "Flanking Maneuver and Converge" and \
                                (
                                        self.status == UAVStatus.APPROACHING_TARGET or self.status == UAVStatus.ENGAGING_TARGET):
                            dist_to_GBAD_center = np.linalg.norm(self.pos_xy - GBAD_TARGET_POS_3D[:2])
                            if dist_to_GBAD_center < (GBAD_EXCLUSION_RADIUS + 50.0):
                                is_legit_GBAD_approach = True
                    if is_legit_GBAD_approach:
                        continue  # Skip NFZ check for this specific GBAD zone if legitimate approach.

                is_violating_this_nfz = False
                proactive_trigger_dist_nfz = (nfz_def["params"]["radius"] if nfz_def["type"] == "circle" else
                                              nfz_def["params"]["effective_radius"]) + nfz_def["anticipation_margin"]

                # Check for intersection with circular or polygonal NFZs.
                if nfz_def["type"] == "circle":
                    dist_to_nfz_center = np.linalg.norm(self.pos_xy - nfz_def["params"]["center"])
                    dist_next_to_nfz_center = np.linalg.norm(next_pos_estimate_short - nfz_def["params"]["center"])
                    if dist_next_to_nfz_center < proactive_trigger_dist_nfz or dist_to_nfz_center < nfz_def["params"][
                        "radius"]:
                        is_violating_this_nfz = True
                elif nfz_def["type"] == "polygon":
                    if is_point_inside_polygon(next_pos_estimate_short, nfz_def["params"]["vertices"]) or \
                            is_point_inside_polygon(self.pos_xy, nfz_def["params"]["vertices"]):
                        is_violating_this_nfz = True

                # Check if this specific NFZ allows a legitimate post-engage clearing path.
                is_legit_clearance_path = False
                if nfz_def["name"] == "Irregular NFZ 1" and self.status == UAVStatus.POST_ENGAGE_CLEARING:
                    if self.current_waypoint_3d is not None and not is_point_inside_polygon(
                            self.current_waypoint_3d[:2], nfz_def["params"]["vertices"]):
                        is_legit_clearance_path = True

                if is_violating_this_nfz and not is_legit_clearance_path:
                    # If leader violates Irregular NFZ 1, it becomes incapacitated and a new leader is elected.
                    if self.is_leader and nfz_def["name"] == "Irregular NFZ 1":
                        self.must_rtb_after_nfz_violation = True
                        elect_new_leader(uavs, self.id, selection_criteria="closest_to_base")
                        self.status = UAVStatus.AVOIDING_NFZ
                        self.currently_avoiding_nfz_details = nfz_def
                        self.previous_status_before_avoidance = UAVStatus.RETURNING_TO_BASE
                        nfz_violation_detected_this_step = True
                        break  # Only avoid one NFZ at a time
                    else:  # General NFZ avoidance for any UAV.
                        self.previous_status_before_avoidance = self.status
                        self.status = UAVStatus.AVOIDING_NFZ
                        self.currently_avoiding_nfz_details = nfz_def
                        nfz_violation_detected_this_step = True
                        break

        # --- NFZ Avoidance Execution ---
        if self.status == UAVStatus.AVOIDING_NFZ:
            nfz_details = self.currently_avoiding_nfz_details
            avoid_center = nfz_details["params"]["center"] if nfz_details["type"] == "circle" else \
                nfz_details["params"]["centroid"]
            vec_from_nfz_center = self.pos_xy - avoid_center

            current_dist_to_nfz_center = np.linalg.norm(vec_from_nfz_center)
            target_clear_radius_nfz = (nfz_details["params"]["radius"] if nfz_details["type"] == "circle" else
                                       nfz_details["params"]["effective_radius"]) + \
                                      nfz_details["anticipation_margin"] + WP_PROXIMITY_XY * 0.5

            # Calculate a heading away from the NFZ center.
            if np.linalg.norm(vec_from_nfz_center) > 1e-5:
                target_wp_xy_for_heading = self.pos_xy + (
                            vec_from_nfz_center / np.linalg.norm(vec_from_nfz_center)) * 50
            else:  # If UAV is exactly at the NFZ center (unlikely), move straight.
                target_wp_xy_for_heading = self.pos_xy + np.array([math.cos(self.hdg_rad), math.sin(self.hdg_rad)]) * 50
            d_alt = self.alt  # Maintain current altitude during avoidance
            d_spd_xy = UAV_MIN_AIRBORNE_SPEED * 1.5  # Increase speed to quickly exit NFZ.

            # Check if UAV has cleared the NFZ.
            is_outside_nfz_boundary = False
            if nfz_details["type"] == "circle":
                is_outside_nfz_boundary = current_dist_to_nfz_center > target_clear_radius_nfz
            elif nfz_details["type"] == "polygon":
                is_outside_nfz_boundary = not is_point_inside_polygon(self.pos_xy,
                                                                      nfz_details["params"]["vertices"]) and \
                                          current_dist_to_nfz_center > target_clear_radius_nfz

            if is_outside_nfz_boundary:
                # If leader was incapacitated due to NFZ violation, it's forced to RTB.
                if self.must_rtb_after_nfz_violation:
                    self.status = UAVStatus.RETURNING_TO_BASE
                    self.current_waypoint_3d = RETURN_BASE_POS_3D
                    self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.75
                    self.is_on_detour_to_base = False
                    self.must_rtb_after_nfz_violation = False
                # If follower's leader is RTB, the follower also transitions to RTB after avoidance.
                elif not self.is_leader and self.leader_ref and self.leader_ref.is_active and self.leader_ref.status == UAVStatus.RETURNING_TO_BASE and self.status != UAVStatus.RETURNING_TO_BASE:
                    self.status = UAVStatus.RETURNING_TO_BASE
                    self.current_waypoint_3d = RETURN_BASE_POS_3D
                    self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.75
                    self.is_on_detour_to_base = False
                else:  # Otherwise, revert to previous status.
                    restored_status = self.previous_status_before_avoidance if self.previous_status_before_avoidance else UAVStatus.FLYING_TO_WAYPOINT
                    if restored_status == UAVStatus.RETURNING_TO_BASE and not self.is_on_detour_to_base:  # Special handling for RTB path planning after detour.
                        self.status = UAVStatus.RETURNING_TO_BASE
                        self.current_waypoint_3d = RETURN_BASE_POS_3D
                        self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.75
                        self.previous_status_before_avoidance = None
                    else:  # Revert to the status before avoidance.
                        self.status = restored_status
                        if self.status != UAVStatus.RETURNING_TO_BASE: self.is_on_detour_to_base = False
                        self.previous_status_before_avoidance = None
                        self.currently_avoiding_nfz_details = None

        # --- Main State Machine (runs if not in AVOIDING_NFZ state) ---
        if self.status != UAVStatus.AVOIDING_NFZ:
            # --- LEADER DIRECT ATTACK Strategy ---
            if SIM_ATTACK_STRATEGY == "Leader Direct Attack":
                if self.is_leader:  # Leader's behavior
                    if self.status == UAVStatus.FLYING_TO_WAYPOINT:  # Initial state, transition to approaching target.
                        self.status = UAVStatus.APPROACHING_TARGET
                        self.current_waypoint_3d = LEADER_ATTACK_RELEASE_POINT_3D
                        self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.8
                    elif self.status == UAVStatus.APPROACHING_TARGET:  # Flying to release point.
                        target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                        d_alt = self.current_waypoint_3d[2]
                        d_spd_xy = UAV_MAX_SPEED * 0.8
                        if np.linalg.norm(
                                self.pos_xy - self.current_waypoint_3d[:2]) < WP_PROXIMITY_XY:  # Reached release point.
                            self.status = UAVStatus.ENGAGING_TARGET
                            self.action_timer = 3.0  # Engage for 3 seconds
                            GBAD_vulnerable_event_time = simulation_time + 0.1  # GBAD becomes vulnerable
                            self.missile_dropped_display_timer = 1.0  # Display missile dropped info.
                            self.commanded_target_speed_xy = UAV_MIN_AIRBORNE_SPEED * 1.2  # Slow down after attack.
                    elif self.status == UAVStatus.ENGAGING_TARGET:  # Firing phase.
                        self.action_timer -= time_step
                        if self.action_timer <= 0:  # Firing complete, start evasion.
                            vec_from_GBAD = self.pos_xy - GBAD_TARGET_POS_3D[:2]
                            self.evasion_heading_rad = math.atan2(vec_from_GBAD[1], vec_from_GBAD[0]) if np.linalg.norm(
                                vec_from_GBAD) > 1e-5 else self.hdg_rad
                            self.status = UAVStatus.EVADING
                            self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.8
                            self.action_timer = LEADER_EVASION_DURATION  # Evade for a set duration.
                        target_wp_xy_for_heading = self.pos_xy + np.array(
                            [math.cos(self.hdg_rad), math.sin(self.hdg_rad)]) * 10  # Continue current heading slightly.
                        d_alt = self.alt
                        d_spd_xy = self.commanded_target_speed_xy
                    elif self.status == UAVStatus.EVADING:  # Evasion phase.
                        d_hdg_rad = self.evasion_heading_rad
                        d_alt = self.alt + 1.0 * time_step  # Slight climb during evasion.
                        d_spd_xy = self.commanded_target_speed_xy
                        self.action_timer -= time_step
                        target_wp_xy_for_heading = None  # No specific waypoint, just follow evasion heading.
                        if self.action_timer <= 0:  # Evasion complete, return to base.
                            self.status = UAVStatus.RETURNING_TO_BASE
                            self.current_waypoint_3d = RETURN_BASE_POS_3D
                            self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.75
                            self.is_on_detour_to_base = False
                    elif self.status == UAVStatus.RETURNING_TO_BASE:  # Returning to base.
                        # Intelligent NFZ avoidance for RTB path planning.
                        if SIM_INCLUDE_NFZ and NFZ_DEFINITIONS and not self.is_on_detour_to_base:
                            direct_path_start = self.pos_xy
                            direct_path_end = RETURN_BASE_POS_3D[:2]

                            # Check for "Irregular NFZ 1" intersection on direct path.
                            nfz_to_check = next((nfz for nfz in NFZ_DEFINITIONS if nfz["name"] == "Irregular NFZ 1"),
                                                None)
                            if nfz_to_check:
                                path_intersects = False
                                if nfz_to_check["type"] == "circle":
                                    path_intersects = line_intersects_circle(direct_path_start, direct_path_end,
                                                                             nfz_to_check["params"]["center"],
                                                                             nfz_to_check["params"]["radius"] +
                                                                             nfz_to_check["anticipation_margin"])
                                elif nfz_to_check["type"] == "polygon":
                                    path_intersects = line_intersects_polygon(direct_path_start, direct_path_end,
                                                                              nfz_to_check["params"]["vertices"]) or \
                                                      is_point_inside_polygon(self.pos_xy,
                                                                              nfz_to_check["params"]["vertices"])

                                if path_intersects:  # If path intersects NFZ, plan a detour.
                                    self.is_on_detour_to_base = True
                                    nfz_center = nfz_to_check["params"]["center"] if nfz_to_check[
                                                                                         "type"] == "circle" else \
                                        nfz_to_check["params"]["centroid"]
                                    vec_uav_to_nfz_center = nfz_center - self.pos_xy

                                    if np.linalg.norm(vec_uav_to_nfz_center) > 1e-5:
                                        norm_vec_nfz_to_uav = vec_uav_to_nfz_center / np.linalg.norm(
                                            vec_uav_to_nfz_center)
                                    else:
                                        norm_vec_nfz_to_uav = np.array([1.0, 0.0])

                                    # Calculate two perpendicular vectors to detour around the NFZ.
                                    perp_vec1 = np.array([-norm_vec_nfz_to_uav[1], norm_vec_nfz_to_uav[0]])
                                    perp_vec2 = np.array([norm_vec_nfz_to_uav[1], -norm_vec_nfz_to_uav[0]])

                                    effective_nfz_radius = (
                                        nfz_to_check["params"]["radius"] if nfz_to_check["type"] == "circle" else
                                        nfz_to_check["params"]["effective_radius"])

                                    detour_wp1_xy = nfz_center + perp_vec1 * (
                                                effective_nfz_radius + nfz_to_check["anticipation_margin"] + 20.0)
                                    detour_wp2_xy = nfz_center + perp_vec2 * (
                                                effective_nfz_radius + nfz_to_check["anticipation_margin"] + 20.0)

                                    # Choose the detour point that is closer to the final base.
                                    dist_to_base_from_wp1 = np.linalg.norm(detour_wp1_xy - RETURN_BASE_POS_3D[:2])
                                    dist_to_base_from_wp2 = np.linalg.norm(detour_wp2_xy - RETURN_BASE_POS_3D[:2])

                                    if dist_to_base_from_wp1 < dist_to_base_from_wp2:
                                        chosen_detour_xy = detour_wp1_xy
                                    else:
                                        chosen_detour_xy = detour_wp2_xy

                                    self.current_waypoint_3d = np.array(
                                        [chosen_detour_xy[0], chosen_detour_xy[1], RETURN_BASE_POS_3D[2] + 20])
                                    self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.6
                            else:  # If NFZ is not defined, go straight to base.
                                self.current_waypoint_3d = RETURN_BASE_POS_3D
                                self.is_on_detour_to_base = False

                        if self.current_waypoint_3d is None:  # Fallback if waypoint is somehow unset.
                            self.current_waypoint_3d = RETURN_BASE_POS_3D
                            self.is_on_detour_to_base = False

                        target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                        d_alt = self.current_waypoint_3d[2]
                        d_spd_xy = self.commanded_target_speed_xy

                        if np.linalg.norm(self.pos_xy - self.current_waypoint_3d[
                                                        :2]) < WP_PROXIMITY_XY:  # Reached current RTB waypoint.
                            if self.is_on_detour_to_base:  # If it was a detour waypoint, now target the actual base.
                                self.current_waypoint_3d = RETURN_BASE_POS_3D
                                self.is_on_detour_to_base = False
                                target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                                d_alt = self.current_waypoint_3d[2]
                                self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.75
                            else:  # Reached final base.
                                self.status = UAVStatus.LANDED
                                self.commanded_target_speed_xy = 0
                                d_spd_xy = 0
                                d_alt = RETURN_BASE_POS_3D[2]
                                if self.is_leader:  # If the leader lands, elect a new leader.
                                    elect_new_leader(uavs, self.id, selection_criteria="closest_to_base")
                else:  # Follower's behavior for Leader Direct Attack.
                    if self.status == UAVStatus.FLYING_TO_WAYPOINT:  # Initial state, transition to waiting.
                        self.status = UAVStatus.WAITING_FOR_ENGAGE_ORDER
                        self.commanded_target_speed_xy = self.leader_ref.commanded_target_speed_xy if self.leader_ref and self.leader_ref.is_active else UAV_MIN_AIRBORNE_SPEED
                    elif self.status == UAVStatus.WAITING_FOR_ENGAGE_ORDER:  # Follower stays in formation, waiting for GBAD counter-attack.
                        proactive_GBAD_avoid_trigger_dist = GBAD_EXCLUSION_RADIUS + GBAD_AVOID_ANTICIPATION_MARGIN
                        if GBAD_lock_on_start_time > 0 and simulation_time >= GBAD_lock_on_start_time and simulation_time <= GBAD_lock_on_start_time + GBAD_LOCK_ON_DURATION:
                            # If GBAD is counter-attacking, check if follower needs to avoid.
                            if np.linalg.norm(self.pos_xy - GBAD_TARGET_POS_3D[:2]) < proactive_GBAD_avoid_trigger_dist:
                                if SIM_INCLUDE_NFZ and NFZ_DEFINITIONS and self.status != UAVStatus.AVOIDING_NFZ:
                                    GBAD_nfz_def = next((nfz for nfz in NFZ_DEFINITIONS if nfz["name"] == "GBAD Zone"),
                                                        None)
                                    if GBAD_nfz_def:
                                        self.previous_status_before_avoidance = UAVStatus.WAITING_FOR_ENGAGE_ORDER
                                        self.status = UAVStatus.AVOIDING_NFZ
                                        self.currently_avoiding_nfz_details = GBAD_nfz_def
                            else:  # If safe, engage perimeter.
                                self.status = UAVStatus.ENGAGING_TARGET
                                vec = GBAD_TARGET_POS_3D[:2] - self.pos_xy
                                norm_vec = vec / (np.linalg.norm(vec) + 1e-6)
                                wp_xy = GBAD_TARGET_POS_3D[:2] - norm_vec * (
                                            GBAD_EXCLUSION_RADIUS + ATTACK_POINT_BUFFER)
                                self.current_waypoint_3d = np.array([wp_xy[0], wp_xy[1], GBAD_TARGET_POS_3D[2]])
                                self.commanded_target_speed_xy = ATTACK_RUN_SPEED
                                self.action_timer = FOLLOWER_ENGAGE_ACTION_TIME  # Engage for a set duration.
                                self.missile_dropped_display_timer = 1.0
                        else:  # Maintain formation if no GBAD counter-attack.
                            if self.leader_ref and self.leader_ref.is_active:
                                des_pos_xy, des_hdg, des_spd, des_alt = self._calculate_follower_target_state()
                                self.current_waypoint_3d = np.array([des_pos_xy[0], des_pos_xy[1], des_alt])
                                target_wp_xy_for_heading = des_pos_xy
                                d_hdg_rad = des_hdg
                                d_spd_xy = des_spd
                                d_alt = des_alt
                            else:  # If leader is inactive, just hold position.
                                target_wp_xy_for_heading = self.pos_xy
                                d_spd_xy = UAV_MIN_AIRBORNE_SPEED
                    elif self.status == UAVStatus.ENGAGING_TARGET:  # Follower engaging phase.
                        if self.current_waypoint_3d is None:  # Fallback if waypoint is unset.
                            self.status = UAVStatus.RETURNING_TO_BASE
                            self.current_waypoint_3d = RETURN_BASE_POS_3D
                        else:  # Fly towards engagement point.
                            target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                            d_alt = self.current_waypoint_3d[2]
                            d_spd_xy = ATTACK_RUN_SPEED
                        if np.linalg.norm(self.pos_xy - self.current_waypoint_3d[
                                                        :2]) < WP_PROXIMITY_XY * 1.5:  # Near engagement point.
                            if self.action_timer <= 0 and self.action_timer > -TIME_STEP:  # Initial engagement.
                                self.action_timer = FOLLOWER_ENGAGE_ACTION_TIME
                                self.missile_dropped_display_timer = 1.0
                            if self.action_timer > 0:  # Continue engaging.
                                self.action_timer -= time_step
                            target_wp_xy_for_heading = self.pos_xy + np.array(
                                [math.cos(self.hdg_rad), math.sin(self.hdg_rad)]) * 10  # Hold position slightly.
                        if self.action_timer <= 0:  # Engagement complete, start clearing maneuver.
                            self.status = UAVStatus.POST_ENGAGE_CLEARING
                            vec_G_A = self.pos_xy - GBAD_TARGET_POS_3D[:2]
                            norm_G_A = vec_G_A / (np.linalg.norm(vec_G_A) + 1e-6) if np.linalg.norm(
                                vec_G_A) > 1e-5 else np.array([1.0, 0.0])
                            vec_G_B = RETURN_BASE_POS_3D[:2] - GBAD_TARGET_POS_3D[:2]
                            norm_G_B = vec_G_B / (np.linalg.norm(vec_G_B) + 1e-6) if np.linalg.norm(
                                vec_G_B) > 1e-5 else np.array([1.0, 0.0])
                            cross_prod_z = norm_G_A[0] * norm_G_B[1] - norm_G_A[1] * norm_G_B[0]
                            angle_G_A = math.atan2(norm_G_A[1], norm_G_A[0])
                            offset = math.pi / 2.2  # Angle offset for clearing maneuver.
                            target_angle = angle_G_A - offset if cross_prod_z > 0 else angle_G_A + offset
                            clear_rad = GBAD_EXCLUSION_RADIUS + 45.0
                            clear_wp = GBAD_TARGET_POS_3D[:2] + np.array(
                                [math.cos(target_angle), math.sin(target_angle)]) * clear_rad
                            self.current_waypoint_3d = np.array([clear_wp[0], clear_wp[1], RETURN_BASE_POS_3D[2] + 10])
                            self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.7
                            self.is_on_detour_to_base = False
                    elif self.status == UAVStatus.POST_ENGAGE_CLEARING:  # Clearing maneuver after engagement.
                        if self.current_waypoint_3d is None:  # Fallback if waypoint unset.
                            self.status = UAVStatus.RETURNING_TO_BASE
                            self.current_waypoint_3d = RETURN_BASE_POS_3D
                        else:  # Fly towards clearing point.
                            target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                            d_alt = self.current_waypoint_3d[2]
                            d_spd_xy = self.commanded_target_speed_xy
                        # Transition to RTB after short clearing or reaching clearing point.
                        if self.action_timer <= 0 or np.linalg.norm(
                                self.pos_xy - self.current_waypoint_3d[:2]) < WP_PROXIMITY_XY * 2:
                            self.status = UAVStatus.RETURNING_TO_BASE
                            self.current_waypoint_3d = RETURN_BASE_POS_3D
                            self.is_on_detour_to_base = False
                    elif self.status == UAVStatus.RETURNING_TO_BASE:  # Returning to base (same logic as leader).
                        if SIM_INCLUDE_NFZ and NFZ_DEFINITIONS and not self.is_on_detour_to_base:
                            direct_path_start = self.pos_xy
                            direct_path_end = RETURN_BASE_POS_3D[:2]

                            nfz_to_check = next((nfz for nfz in NFZ_DEFINITIONS if nfz["name"] == "Irregular NFZ 1"),
                                                None)
                            if nfz_to_check:
                                path_intersects = False
                                if nfz_to_check["type"] == "circle":
                                    path_intersects = line_intersects_circle(direct_path_start, direct_path_end,
                                                                             nfz_to_check["params"]["center"],
                                                                             nfz_to_check["params"]["radius"] +
                                                                             nfz_to_check["anticipation_margin"])
                                elif nfz_to_check["type"] == "polygon":
                                    path_intersects = line_intersects_polygon(direct_path_start, direct_path_end,
                                                                              nfz_to_check["params"]["vertices"]) or \
                                                      is_point_inside_polygon(self.pos_xy,
                                                                              nfz_to_check["params"]["vertices"])

                                if path_intersects:
                                    self.is_on_detour_to_base = True
                                    nfz_center = nfz_to_check["params"]["center"] if nfz_to_check[
                                                                                         "type"] == "circle" else \
                                        nfz_to_check["params"]["centroid"]
                                    vec_uav_to_nfz_center = nfz_center - self.pos_xy

                                    if np.linalg.norm(vec_uav_to_nfz_center) > 1e-5:
                                        norm_vec_nfz_to_uav = vec_uav_to_nfz_center / np.linalg.norm(
                                            vec_uav_to_nfz_center)
                                    else:
                                        norm_vec_nfz_to_uav = np.array([1.0, 0.0])

                                    perp_vec1 = np.array([-norm_vec_nfz_to_uav[1], norm_vec_nfz_to_uav[0]])
                                    perp_vec2 = np.array([norm_vec_nfz_to_uav[1], -norm_vec_nfz_to_uav[0]])

                                    effective_nfz_radius = (
                                        nfz_to_check["params"]["radius"] if nfz_to_check["type"] == "circle" else
                                        nfz_to_check["params"]["effective_radius"])

                                    detour_wp1_xy = nfz_center + perp_vec1 * (
                                                effective_nfz_radius + nfz_to_check["anticipation_margin"] + 20.0)
                                    detour_wp2_xy = nfz_center + perp_vec2 * (
                                                effective_nfz_radius + nfz_to_check["anticipation_margin"] + 20.0)

                                    dist_to_base_from_wp1 = np.linalg.norm(detour_wp1_xy - RETURN_BASE_POS_3D[:2])
                                    dist_to_base_from_wp2 = np.linalg.norm(detour_wp2_xy - RETURN_BASE_POS_3D[:2])

                                    if dist_to_base_from_wp1 < dist_to_base_from_wp2:
                                        chosen_detour_xy = detour_wp1_xy
                                    else:
                                        chosen_detour_xy = detour_wp2_xy

                                    self.current_waypoint_3d = np.array(
                                        [chosen_detour_xy[0], chosen_detour_xy[1], RETURN_BASE_POS_3D[2] + 20])
                                    self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.6
                            else:
                                self.current_waypoint_3d = RETURN_BASE_POS_3D
                                self.is_on_detour_to_base = False

                        if self.current_waypoint_3d is None:
                            self.current_waypoint_3d = RETURN_BASE_POS_3D
                            self.is_on_detour_to_base = False

                        target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                        d_alt = self.current_waypoint_3d[2]
                        d_spd_xy = self.commanded_target_speed_xy

                        if np.linalg.norm(self.pos_xy - self.current_waypoint_3d[:2]) < WP_PROXIMITY_XY:
                            if self.is_on_detour_to_base:
                                self.current_waypoint_3d = RETURN_BASE_POS_3D
                                self.is_on_detour_to_base = False
                                target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                                d_alt = self.current_waypoint_3d[2]
                                self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.75
                            else:
                                self.status = UAVStatus.LANDED
                                self.commanded_target_speed_xy = 0
                                d_spd_xy = 0
                                d_alt = RETURN_BASE_POS_3D[2]

            # --- FLANKING MANEUVER AND CONVERGE Strategy ---
            elif SIM_ATTACK_STRATEGY == "Flanking Maneuver and Converge":
                # Defines specific waypoints and behaviors for the flanking strategy.
                FLANK_DISTANCE = 100.0
                CONVERGE_RADIUS = 30.0
                ENGAGE_DURATION_FLANK = 3.0
                POST_ENGAGE_CLEARING_DURATION = 2.0

                # Fixed flanking waypoints.
                FLANK_WP_1 = np.array(
                    [GBAD_TARGET_POS_3D[0] - 100, GBAD_TARGET_POS_3D[1] + FLANK_DISTANCE, GBAD_TARGET_POS_3D[2] + 30])
                FLANK_WP_2 = np.array(
                    [GBAD_TARGET_POS_3D[0] - 100, GBAD_TARGET_POS_3D[1] - FLANK_DISTANCE, GBAD_TARGET_POS_3D[2] + 30])

                # Point where UAVs converge after flanking.
                CONVERGE_POINT = np.array(
                    [GBAD_TARGET_POS_3D[0] + CONVERGE_RADIUS, GBAD_TARGET_POS_3D[1], GBAD_TARGET_POS_3D[2] + 10])

                if self.status == UAVStatus.FLYING_TO_WAYPOINT:  # Initial state, assign flanking waypoints.
                    if self.id % 2 == 0:  # Even IDs go to flank 1
                        self.current_waypoint_3d = FLANK_WP_1
                    else:  # Odd IDs go to flank 2
                        self.current_waypoint_3d = FLANK_WP_2
                    self.status = UAVStatus.APPROACHING_TARGET
                    self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.7
                elif self.status == UAVStatus.APPROACHING_TARGET:  # Flying to flanking waypoint.
                    target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                    d_alt = self.current_waypoint_3d[2]
                    d_spd_xy = UAV_MAX_SPEED * 0.7

                    if np.linalg.norm(self.pos_xy - self.current_waypoint_3d[
                                                    :2]) < WP_PROXIMITY_XY:  # Reached flanking waypoint, proceed to converge.
                        self.current_waypoint_3d = CONVERGE_POINT
                        self.status = UAVStatus.ENGAGING_TARGET  # Transition to engage.
                        self.commanded_target_speed_xy = ATTACK_RUN_SPEED
                        self.action_timer = ENGAGE_DURATION_FLANK
                        GBAD_vulnerable_event_time = simulation_time + 0.1
                        self.missile_dropped_display_timer = 1.0
                elif self.status == UAVStatus.ENGAGING_TARGET:  # Converging and engaging phase.
                    if self.current_waypoint_3d is None:  # Fallback if waypoint unset.
                        self.status = UAVStatus.RETURNING_TO_BASE
                        self.current_waypoint_3d = RETURN_BASE_POS_3D
                    else:  # Fly towards converge point.
                        target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                        d_alt = self.current_waypoint_3d[2]
                        d_spd_xy = ATTACK_RUN_SPEED
                        self.action_timer -= time_step

                        if np.linalg.norm(self.pos_xy - self.current_waypoint_3d[
                                                        :2]) < WP_PROXIMITY_XY * 1.5 and self.action_timer > 0:
                            target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                        else:  # Converged/engaged, start clearing.
                            if self.action_timer <= 0:
                                self.status = UAVStatus.POST_ENGAGE_CLEARING
                                self.action_timer = POST_ENGAGE_CLEARING_DURATION
                                # Calculate a clear path away from GBAD, towards base.
                                vec_from_GBAD = self.pos_xy - GBAD_TARGET_POS_3D[:2]
                                if np.linalg.norm(vec_from_GBAD) > 1e-5:
                                    direction_away_from_GBAD = vec_from_GBAD / np.linalg.norm(vec_from_GBAD)
                                    direction_towards_base = (RETURN_BASE_POS_3D[:2] - self.pos_xy) / (
                                                np.linalg.norm(RETURN_BASE_POS_3D[:2] - self.pos_xy) + 1e-6)
                                    clear_direction_vec = (direction_away_from_GBAD + direction_towards_base) / 2
                                    if np.linalg.norm(clear_direction_vec) > 1e-5:
                                        clear_direction_vec = clear_direction_vec / np.linalg.norm(clear_direction_vec)
                                    else:
                                        clear_direction_vec = direction_away_from_GBAD
                                else:
                                    clear_direction_vec = (RETURN_BASE_POS_3D[:2] - GBAD_TARGET_POS_3D[:2]) / (
                                                np.linalg.norm(RETURN_BASE_POS_3D[:2] - GBAD_TARGET_POS_3D[:2]) + 1e-6)

                                clear_wp = self.pos_xy + clear_direction_vec * 50.0
                                self.current_waypoint_3d = np.array(
                                    [clear_wp[0], clear_wp[1], RETURN_BASE_POS_3D[2] + 10])
                                self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.7
                                self.is_on_detour_to_base = False
                elif self.status == UAVStatus.POST_ENGAGE_CLEARING:  # Clearing maneuver.
                    self.action_timer -= time_step
                    if self.current_waypoint_3d is None:  # Fallback if waypoint unset.
                        self.status = UAVStatus.RETURNING_TO_BASE
                        self.current_waypoint_3d = RETURN_BASE_POS_3D
                    else:
                        target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                        d_alt = self.current_waypoint_3d[2]
                        d_spd_xy = self.commanded_target_speed_xy
                    # Transition to RTB after short clearing or reaching clearing point.
                    if self.action_timer <= 0 or np.linalg.norm(
                            self.pos_xy - self.current_waypoint_3d[:2]) < WP_PROXIMITY_XY * 2:
                        self.status = UAVStatus.RETURNING_TO_BASE
                        self.current_waypoint_3d = RETURN_BASE_POS_3D
                        self.is_on_detour_to_base = False
                elif self.status == UAVStatus.RETURNING_TO_BASE:  # Returning to base (same logic as Leader Direct Attack).
                    if SIM_INCLUDE_NFZ and NFZ_DEFINITIONS and not self.is_on_detour_to_base:
                        direct_path_start = self.pos_xy
                        direct_path_end = RETURN_BASE_POS_3D[:2]

                        nfz_to_check = next((nfz for nfz in NFZ_DEFINITIONS if nfz["name"] == "Irregular NFZ 1"), None)
                        if nfz_to_check:
                            path_intersects = False
                            if nfz_to_check["type"] == "circle":
                                path_intersects = line_intersects_circle(direct_path_start, direct_path_end,
                                                                         nfz_to_check["params"]["center"],
                                                                         nfz_to_check["params"]["radius"] +
                                                                         nfz_to_check["anticipation_margin"])
                            elif nfz_to_check["type"] == "polygon":
                                path_intersects = line_intersects_polygon(direct_path_start, direct_path_end,
                                                                          nfz_to_check["params"]["vertices"]) or \
                                                  is_point_inside_polygon(self.pos_xy,
                                                                          nfz_to_check["params"]["vertices"])

                            if path_intersects:
                                self.is_on_detour_to_base = True
                                nfz_center = nfz_to_check["params"]["center"] if nfz_to_check["type"] == "circle" else \
                                    nfz_to_check["params"]["centroid"]
                                vec_uav_to_nfz_center = nfz_center - self.pos_xy

                                if np.linalg.norm(vec_uav_to_nfz_center) > 1e-5:
                                    norm_vec_nfz_to_uav = vec_uav_to_nfz_center / np.linalg.norm(vec_uav_to_nfz_center)
                                else:
                                    norm_vec_nfz_to_uav = np.array([1.0, 0.0])

                                perp_vec1 = np.array([-norm_vec_nfz_to_uav[1], norm_vec_nfz_to_uav[0]])
                                perp_vec2 = np.array([norm_vec_nfz_to_uav[1], -norm_vec_nfz_to_uav[0]])

                                effective_nfz_radius = (
                                    nfz_to_check["params"]["radius"] if nfz_to_check["type"] == "circle" else
                                    nfz_to_check["params"]["effective_radius"])

                                detour_wp1_xy = nfz_center + perp_vec1 * (
                                            effective_nfz_radius + nfz_to_check["anticipation_margin"] + 20.0)
                                detour_wp2_xy = nfz_center + perp_vec2 * (
                                            effective_nfz_radius + nfz_to_check["anticipation_margin"] + 20.0)

                                dist_to_base_from_wp1 = np.linalg.norm(detour_wp1_xy - RETURN_BASE_POS_3D[:2])
                                dist_to_base_from_wp2 = np.linalg.norm(detour_wp2_xy - RETURN_BASE_POS_3D[:2])

                                if dist_to_base_from_wp1 < dist_to_base_from_wp2:
                                    chosen_detour_xy = detour_wp1_xy
                                else:
                                    chosen_detour_xy = detour_wp2_xy

                                self.current_waypoint_3d = np.array(
                                    [chosen_detour_xy[0], chosen_detour_xy[1], RETURN_BASE_POS_3D[2] + 20])
                                self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.6
                            else:
                                self.current_waypoint_3d = RETURN_BASE_POS_3D
                                self.is_on_detour_to_base = False

                        if self.current_waypoint_3d is None:
                            self.current_waypoint_3d = RETURN_BASE_POS_3D
                            self.is_on_detour_to_base = False

                        target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                        d_alt = self.current_waypoint_3d[2]
                        d_spd_xy = self.commanded_target_speed_xy

                        if np.linalg.norm(self.pos_xy - self.current_waypoint_3d[:2]) < WP_PROXIMITY_XY:
                            if self.is_on_detour_to_base:
                                self.current_waypoint_3d = RETURN_BASE_POS_3D
                                self.is_on_detour_to_base = False
                                target_wp_xy_for_heading = self.current_waypoint_3d[:2]
                                d_alt = self.current_waypoint_3d[2]
                                self.commanded_target_speed_xy = UAV_MAX_SPEED * 0.75
                            else:
                                self.status = UAVStatus.LANDED
                                self.commanded_target_speed_xy = 0
                                d_spd_xy = 0
                                d_alt = RETURN_BASE_POS_3D[2]

        # --- Final Control Application ---
        if self.status == UAVStatus.LANDED:  # If landed, stop all movement and stay at base altitude.
            self.speed_xy = 0
            self.vertical_speed = 0
            self.alt = RETURN_BASE_POS_3D[2]
            self.current_waypoint_3d = None
        elif self.status == UAVStatus.INCAPACITATED:  # If incapacitated, stop all movement.
            self.speed_xy = 0
            self.vertical_speed = 0
        else:  # Apply controls based on calculated desired states.
            # Determine the target waypoint for heading control.
            if target_wp_xy_for_heading is None and self.current_waypoint_3d is not None and \
                    not (self.is_leader and self.status == UAVStatus.EVADING):
                target_wp_xy_for_heading = self.current_waypoint_3d[:2]

            if target_wp_xy_for_heading is not None:
                vec_to_wp = target_wp_xy_for_heading - self.pos_xy
                if np.linalg.norm(vec_to_wp) > WP_PROXIMITY_XY * 0.25:
                    is_direct_nav_state = self.status in [
                        UAVStatus.APPROACHING_TARGET, UAVStatus.ENGAGING_TARGET,
                        UAVStatus.RETURNING_TO_BASE, UAVStatus.AVOIDING_NFZ,
                        UAVStatus.POST_ENGAGE_CLEARING
                    ]
                    # Leaders follow direct heading to waypoints unless evading.
                    if self.is_leader and self.status != UAVStatus.EVADING:
                        d_hdg_rad = math.atan2(vec_to_wp[1], vec_to_wp[0])
                    # Followers follow direct heading in specific direct navigation states or if far from leader.
                    elif not self.is_leader and (is_direct_nav_state or \
                                                 (self.status == UAVStatus.WAITING_FOR_ENGAGE_ORDER and np.linalg.norm(
                                                     vec_to_wp) > WP_PROXIMITY_XY * 2.0)):
                        d_hdg_rad = math.atan2(vec_to_wp[1], vec_to_wp[0])

            self.commanded_target_speed_xy = d_spd_xy  # Update commanded speed
            self._control_heading(d_hdg_rad, time_step)  # Apply heading control
            self._control_speed(self.commanded_target_speed_xy, time_step)  # Apply speed control
            self._control_altitude(d_alt, time_step)  # Apply altitude control
            self.pos_xy += np.array([math.cos(self.hdg_rad), math.sin(
                self.hdg_rad)]) * self.speed_xy * time_step  # Update position based on new speed and heading

            self._add_to_path_history()  # Record new position in path history.


# --- Global Functions ---
def elect_new_leader(uav_list_arg, old_leader_id_val, selection_criteria="lowest_id"):
    # Elects a new leader UAV from active, non-leader UAVs.
    global LEADER_ID, uavs, GBAD_vulnerable_event_time, GBAD_lock_on_start_time, SIM_ATTACK_STRATEGY, NFZ_DEFINITIONS, GBAD_EXCLUSION_RADIUS

    old_leader_obj = None
    if 0 <= old_leader_id_val < len(uav_list_arg):
        old_leader_obj = uav_list_arg[old_leader_id_val]
        if old_leader_obj.is_leader:  # Deactivate the old leader.
            old_leader_obj.is_leader = False
            old_leader_obj.is_active = False
            old_leader_obj.status = UAVStatus.INCAPACITATED
            old_leader_obj.speed_xy = 0
            old_leader_obj.vertical_speed = 0
        else:  # Handle case where provided old_leader_id was not actually the leader.
            pass

    # Find active, non-leader UAVs as candidates.
    candidate_followers = [u for u in uav_list_arg if u.is_active and not u.is_leader and u.id != old_leader_id_val]

    if not candidate_followers:  # If no suitable followers, no new leader is elected.
        LEADER_ID = -1
        for uav_obj in uav_list_arg:
            if uav_obj.is_active: uav_obj.leader_ref = None  # Clear leader references.
        return None

    new_leader_obj = None
    # Select new leader based on criteria.
    if selection_criteria == "lowest_id":
        new_leader_obj = min(candidate_followers, key=lambda u: u.id)
    elif selection_criteria == "closest_to_base":
        new_leader_obj = min(candidate_followers, key=lambda u: np.linalg.norm(u.pos_xy - RETURN_BASE_POS_3D[:2]))
    else:  # Default
        new_leader_obj = min(candidate_followers, key=lambda u: u.id)

    if new_leader_obj:  # If a new leader is found.
        LEADER_ID = new_leader_obj.id  # Update global leader ID.
        new_leader_obj.is_leader = True  # Mark as leader.
        new_leader_obj.leader_ref = None  # A leader has no leader reference.
        new_leader_obj.formation_offset_body = np.array([0.0, 0.0, 0.0])  # Leader has zero offset.

        # Determine new leader's mission status based on old leader's status or current mission phase.
        if old_leader_obj and old_leader_obj.status in [UAVStatus.RETURNING_TO_BASE, UAVStatus.LANDED]:
            if new_leader_obj.status not in [UAVStatus.RETURNING_TO_BASE, UAVStatus.LANDED, UAVStatus.AVOIDING_NFZ]:
                new_leader_obj.status = UAVStatus.RETURNING_TO_BASE
                new_leader_obj.current_waypoint_3d = RETURN_BASE_POS_3D
                new_leader_obj.commanded_target_speed_xy = UAV_MAX_SPEED * 0.75
                new_leader_obj.is_on_detour_to_base = False
        elif GBAD_lock_on_start_time > 0 and simulation_time >= GBAD_lock_on_start_time:
            if SIM_ATTACK_STRATEGY == "Leader Direct Attack":
                if new_leader_obj.status not in [UAVStatus.ENGAGING_TARGET, UAVStatus.POST_ENGAGE_CLEARING,
                                                 UAVStatus.EVADING, UAVStatus.RETURNING_TO_BASE, UAVStatus.LANDED,
                                                 UAVStatus.AVOIDING_NFZ]:
                    if NFZ_DEFINITIONS and any(
                            nfz["name"] == "GBAD Zone" for nfz in NFZ_DEFINITIONS) and np.linalg.norm(
                            new_leader_obj.pos_xy - GBAD_TARGET_POS_3D[:2]) < GBAD_EXCLUSION_RADIUS:
                        new_leader_obj.status = UAVStatus.POST_ENGAGE_CLEARING
                        vec_G_A = new_leader_obj.pos_xy - GBAD_TARGET_POS_3D[:2]
                        norm_G_A = vec_G_A / (np.linalg.norm(vec_G_A) + 1e-6) if np.linalg.norm(
                            vec_G_A) > 1e-5 else np.array([1.0, 0.0])
                        offset = math.pi / 2.2
                        target_angle = math.atan2(norm_G_A[1], norm_G_A[0]) - offset
                        clear_rad = GBAD_EXCLUSION_RADIUS + 45.0
                        clear_wp = GBAD_TARGET_POS_3D[:2] + np.array(
                            [math.cos(target_angle), math.sin(target_angle)]) * clear_rad
                        new_leader_obj.current_waypoint_3d = np.array(
                            [clear_wp[0], clear_wp[1], RETURN_BASE_POS_3D[2] + 10])
                        new_leader_obj.commanded_target_speed_xy = UAV_MAX_SPEED * 0.7
                    else:
                        new_leader_obj.status = UAVStatus.APPROACHING_TARGET
                        new_leader_obj.current_waypoint_3d = LEADER_ATTACK_RELEASE_POINT_3D
                        new_leader_obj.commanded_target_speed_xy = UAV_MAX_SPEED * 0.8
            elif SIM_ATTACK_STRATEGY == "Flanking Maneuver and Converge":
                if new_leader_obj.status not in [UAVStatus.APPROACHING_TARGET, UAVStatus.ENGAGING_TARGET,
                                                 UAVStatus.POST_ENGAGE_CLEARING, UAVStatus.RETURNING_TO_BASE,
                                                 UAVStatus.LANDED, UAVStatus.AVOIDING_NFZ]:
                    FLANK_DISTANCE = 100.0
                    FLANK_WP_1 = np.array([GBAD_TARGET_POS_3D[0] - 100, GBAD_TARGET_POS_3D[1] + FLANK_DISTANCE,
                                           GBAD_TARGET_POS_3D[2] + 30])
                    FLANK_WP_2 = np.array([GBAD_TARGET_POS_3D[0] - 100, GBAD_TARGET_POS_3D[1] - FLANK_DISTANCE,
                                           GBAD_TARGET_POS_3D[2] + 30])

                    if new_leader_obj.id % 2 == 0:
                        new_leader_obj.current_waypoint_3d = FLANK_WP_1
                    else:
                        new_leader_obj.current_waypoint_3d = FLANK_WP_2
                    new_leader_obj.status = UAVStatus.APPROACHING_TARGET
                    new_leader_obj.commanded_target_speed_xy = UAV_MAX_SPEED * 0.7
        else:  # Default behavior if no specific mission phase: start according to strategy.
            if new_leader_obj.status not in [UAVStatus.APPROACHING_TARGET, UAVStatus.ENGAGING_TARGET,
                                             UAVStatus.POST_ENGAGE_CLEARING, UAVStatus.EVADING,
                                             UAVStatus.RETURNING_TO_BASE, UAVStatus.LANDED, UAVStatus.AVOIDING_NFZ]:
                if SIM_ATTACK_STRATEGY == "Leader Direct Attack":
                    new_leader_obj.status = UAVStatus.APPROACHING_TARGET
                    new_leader_obj.current_waypoint_3d = LEADER_ATTACK_RELEASE_POINT_3D
                    new_leader_obj.commanded_target_speed_xy = UAV_MAX_SPEED * 0.8
                elif SIM_ATTACK_STRATEGY == "Flanking Maneuver and Converge":
                    FLANK_DISTANCE = 100.0
                    FLANK_WP_1 = np.array([GBAD_TARGET_POS_3D[0] - 100, GBAD_TARGET_POS_3D[1] + FLANK_DISTANCE,
                                           GBAD_TARGET_POS_3D[2] + 30])
                    FLANK_WP_2 = np.array([GBAD_TARGET_POS_3D[0] - 100, GBAD_TARGET_POS_3D[1] - FLANK_DISTANCE,
                                           GBAD_TARGET_POS_3D[2] + 30])

                    if new_leader_obj.id % 2 == 0:
                        new_leader_obj.current_waypoint_3d = FLANK_WP_1
                    else:
                        new_leader_obj.current_waypoint_3d = FLANK_WP_2
                    new_leader_obj.status = UAVStatus.APPROACHING_TARGET
                    new_leader_obj.commanded_target_speed_xy = UAV_MAX_SPEED * 0.7

        # All other active UAVs now reference the new leader.
        for uav_obj in uav_list_arg:
            if uav_obj.is_active and uav_obj.id != new_leader_obj.id:
                uav_obj.is_leader = False
                uav_obj.set_leader_reference(new_leader_obj)
                if uav_obj.status == UAVStatus.WAITING_FOR_ENGAGE_ORDER:  # Followers waiting for orders should now follow the new leader.
                    pass
        return new_leader_obj
    return None  # No new leader could be elected.


# --- Global State & Waypoints ---
uavs = []  # List to hold all UAV objects
simulation_time = 0.0  # Tracks current simulation time
GBAD_target_uav_id = -1  # ID of the UAV GBAD is currently locked onto (initially -1 for no target)


# --- User Input for Simulation Scenario (Before Initialization) ---
def get_user_scenario_choices():
    # Prompts the user to select mission parameters.
    print("\n--- Attack Strategy Selection ---")
    print("1. Leader Direct Attack (Leader attacks directly, followers support)")
    print("2. Flanking Maneuver and Converge (UAVs approach GBAD from two flanks and converge)")

    while True:
        try:
            choice = int(input("Select attack strategy (1 or 2): "))
            if choice == 1:
                selected_strategy = "Leader Direct Attack"
                break
            elif choice == 2:
                selected_strategy = "Flanking Maneuver and Converge"
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print("\n--- NFZ (No-Fly Zone) Inclusion ---")
    while True:
        nfz_choice = input("Include NFZs in simulation? (yes/no): ").lower()
        if nfz_choice in ['yes', 'y']:
            include_nfz = True
            print("NFZs will be included.")
            break
        elif nfz_choice in ['no', 'n']:
            include_nfz = False
            print("NFZs will NOT be included.")
            break
        else:
            print("Invalid choice. Please enter 'yes' or 'no'.")

    print("\n--- UAV Failure Scenario ---")
    while True:
        failure_choice = input("Simulate UAV failure? (yes/no): ").lower()
        if failure_choice in ['yes', 'y']:
            while True:
                try:
                    failure_time = float(input("Enter failure time in seconds (e.g., 5.0): "))
                    if failure_time >= 0:
                        print(f"UAV failure simulated at {failure_time:.1f} seconds.")
                        break
                    else:
                        print("Failure time must be non-negative.")
                except ValueError:
                    print("Invalid input. Please enter a number for failure time.")
            break
        elif failure_choice in ['no', 'n']:
            failure_time = -1.0  # Sentinel value indicates no failure.
            print("No UAV failure will be simulated.")
            break
        else:
            print("Invalid choice. Please enter 'yes' or 'no'.")

    return selected_strategy, include_nfz, failure_time


# Get user choices for the simulation scenario. This runs once at script start.
SIM_ATTACK_STRATEGY, SIM_INCLUDE_NFZ, SIM_FAILURE_TIME = get_user_scenario_choices()

# --- NFZ Definitions (Conditionally set based on user input) ---
# NFZ_DEFINITIONS is initialized as an empty list and populated based on user choice.
NFZ_DEFINITIONS = []

# GBAD Zone is always defined but only added to NFZ_DEFINITIONS if user chooses to include NFZs.
GBAD_ZONE_DEF_FOR_NFZ_LIST = {
    "name": "GBAD Zone", "type": "circle",
    "params": {"center": GBAD_TARGET_POS_3D[:2], "radius": GBAD_EXCLUSION_RADIUS},
    "anticipation_margin": GBAD_AVOID_ANTICIPATION_MARGIN
}
if SIM_INCLUDE_NFZ:
    NFZ_DEFINITIONS.append(GBAD_ZONE_DEF_FOR_NFZ_LIST)  # Add GBAD as an NFZ if chosen
    # Add other irregular NFZ if chosen.
    NFZ_DEFINITIONS.append({
        "name": "Irregular NFZ 1", "type": "polygon",
        "params": {"vertices": IRREGULAR_NFZ_VERTICES, "centroid": IRREGULAR_NFZ_CENTROID,
                   "effective_radius": IRREGULAR_NFZ_EFFECTIVE_RADIUS},
        "anticipation_margin": POLYGON_NFZ_AVOID_ANTICIPATION_MARGIN
    })
else:
    pass  # NFZ_DEFINITIONS remains empty if not included.


# --- Initialization Function ---
def initialize_simulation():
    # Sets up the initial state of all UAVs and global mission parameters.
    global uavs, simulation_time, GBAD_vulnerable_event_time, GBAD_lock_on_start_time, LEADER_ID, SIM_ATTACK_STRATEGY, GBAD_target_uav_id

    uavs = []  # Clear previous UAVs.
    leader = None
    LEADER_ID = 0  # Reset leader ID.
    GBAD_vulnerable_event_time = -1.0  # Reset GBAD vulnerability.
    GBAD_lock_on_start_time = -1.0  # Reset GBAD lock-on time.
    GBAD_target_uav_id = -1  # Reset GBAD target.

    # Determine initial leader position and heading based on selected strategy.
    initial_leader_pos_xy = np.array([SIM_WIDTH * 0.1, SIM_HEIGHT * 0.5])
    initial_leader_alt = UAV_INITIAL_ALTITUDE

    if SIM_ATTACK_STRATEGY == "Leader Direct Attack":
        vec_to_first_target = LEADER_ATTACK_RELEASE_POINT_3D[:2] - initial_leader_pos_xy
    elif SIM_ATTACK_STRATEGY == "Flanking Maneuver and Converge":
        vec_to_first_target = GBAD_TARGET_POS_3D[:2] - initial_leader_pos_xy
    else:  # Fallback to Leader Direct Attack if strategy is unexpected.
        vec_to_first_target = LEADER_ATTACK_RELEASE_POINT_3D[:2] - initial_leader_pos_xy

    initial_leader_hdg_rad = math.atan2(vec_to_first_target[1], vec_to_first_target[0]) if np.linalg.norm(
        vec_to_first_target) > 1e-5 else 0.0

    # Create UAV objects and place them in initial formation.
    for i in range(NUM_UAVS):
        is_l = (i == LEADER_ID)
        if is_l:  # Leader starts at initial leader position.
            start_pos_xy = initial_leader_pos_xy
            start_alt = initial_leader_alt
            start_hdg = initial_leader_hdg_rad
        else:  # Followers start offset from leader based on formation offsets.
            offset_body = FORMATION_OFFSETS[i]
            offset_along, offset_across = offset_body[0], offset_body[1]
            # Rotate offsets by initial leader heading to get world coordinates.
            rot_off_x = offset_along * math.cos(initial_leader_hdg_rad) - offset_across * math.sin(
                initial_leader_hdg_rad)
            rot_off_y = offset_along * math.sin(initial_leader_hdg_rad) + offset_across * math.cos(
                initial_leader_hdg_rad)
            start_pos_xy = initial_leader_pos_xy + np.array([rot_off_x, rot_off_y])
            start_alt = initial_leader_alt + offset_body[2]
            start_hdg = initial_leader_hdg_rad  # Followers match initial leader heading.

        uav = UAV(i, start_pos_xy, start_alt, start_hdg, is_leader=is_l, formation_offset_body=FORMATION_OFFSETS[i])
        if is_l:
            leader = uav  # Store reference to the leader object.
        uavs.append(uav)

    # Set initial status and speed for all UAVs based on selected attack strategy.
    for i, uav_obj in enumerate(uavs):
        uav_obj.status = UAVStatus.FLYING_TO_WAYPOINT
        uav_obj.commanded_target_speed_xy = UAV_MAX_SPEED * 0.8

        if not uav_obj.is_leader and leader:
            uav_obj.set_leader_reference(leader)  # Followers set their leader reference.

    simulation_time = 0.0  # Reset simulation time.


# Initialize the simulation state before starting the animation.
initialize_simulation()

# --- Matplotlib Visualization Setup ---
fig, ax = plt.subplots(figsize=(12, 12))  # Create the plot figure and axes.

# Visualize GBAD Target (always visible as a marker).
GBAD_marker, = ax.plot([GBAD_TARGET_POS_3D[0]], [GBAD_TARGET_POS_3D[1]], marker='X', linestyle='None', markersize=12,
                       color='darkred', label="GBAD Target", alpha=0.8)

# Visualize GBAD Exclusion Zone as a translucent circle patch (always visible).
GBAD_exclusion_patch = patches.Circle(GBAD_TARGET_POS_3D[:2], GBAD_EXCLUSION_RADIUS,
                                      edgecolor='red', facecolor='red', alpha=0.1, linestyle='-',
                                      label="GBAD Exclusion Zone")
ax.add_patch(GBAD_exclusion_patch)

# Visualize other NFZs (only if included by user choice).
nfz_patches = []
if SIM_INCLUDE_NFZ:
    for nfz_def in NFZ_DEFINITIONS:
        if nfz_def["name"] != "GBAD Zone":  # GBAD Zone is drawn separately.
            if nfz_def["type"] == "circle":
                patch = patches.Circle(nfz_def["params"]["center"], nfz_def["params"]["radius"],
                                       edgecolor='magenta', facecolor='magenta', alpha=0.15,
                                       label=f"{nfz_def['name']} (Actual)")
                ax.add_patch(patch)
                nfz_patches.append(patch)
            elif nfz_def["type"] == "polygon":
                patch = patches.Polygon(nfz_def["params"]["vertices"], closed=True,
                                        edgecolor='brown', facecolor='tan', alpha=0.3, label=nfz_def['name'])
                ax.add_patch(patch)
                nfz_patches.append(patch)

# Setup artists for each UAV for visualization.
uav_artists = []
uav_altitude_texts = []
uav_path_lines = []
uav_direction_lines = []
for i in range(NUM_UAVS):
    marker_char = '^' if uavs[i].is_leader else 'v'  # Leader is a triangle, followers are inverted triangles.
    color = 'red' if uavs[i].is_leader else ('blue' if i % 2 == 1 else 'deepskyblue')  # Assign colors to UAVs.

    artist, = ax.plot([], [], marker=marker_char, markersize=10 if uavs[i].is_leader else 8,
                      color=color, label=f"UAV {i}{' (L)' if uavs[i].is_leader else F' (F{i})'}")
    uav_artists.append(artist)
    alt_text = ax.text(0, 0, "", fontsize=7, ha='center', va='bottom', color='black')
    uav_altitude_texts.append(alt_text)
    path_line, = ax.plot([], [], '--', lw=0.8, color=color, alpha=0.6)
    uav_path_lines.append(path_line)
    dir_line, = ax.plot([], [], '-', lw=1.2, color=color, alpha=0.8)
    uav_direction_lines.append(dir_line)

# Visualize the Return Base point.
base_marker, = ax.plot([RETURN_BASE_POS_3D[0]], [RETURN_BASE_POS_3D[1]], marker='s', linestyle='None', markersize=10,
                       color='green', label="Base Point", alpha=0.7)
# Line to show GBAD's lock-on target.
GBAD_lock_on_line, = ax.plot([], [], '--', lw=1.5, color='yellow', alpha=0.9, label="GBAD Lock-On")

# Set plot limits, aspect ratio, legend, labels, and grid.
ax.set_xlim(0, SIM_WIDTH)
ax.set_ylim(0, SIM_HEIGHT)
ax.set_aspect('equal')
ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1.01))
title_text = ax.set_title("")
plt.xlabel("X Coordinate (m)")
plt.ylabel("Y Coordinate (m)")
plt.grid(True, linestyle=':', alpha=0.7)
plt.subplots_adjust(right=0.70)  # Adjust layout to make space for legend.


# --- Simulation Update Function (The main loop) ---
def update_simulation(frame_num):
    # This function is called repeatedly by the animation to update the simulation state and visualization.
    global simulation_time, GBAD_lock_on_start_time, uavs, LEADER_ID, SIM_FAILURE_TIME, GBAD_target_uav_id, GBAD_LOCK_ON_DURATION, GBAD_LOCK_ON_DELAY, GBAD_LOCK_ON_RANGE

    # Get a reference to the current leader UAV.
    current_leader_obj = uavs[LEADER_ID] if 0 <= LEADER_ID < len(uavs) and uavs[LEADER_ID].is_active else None

    # --- UAV FAILURE TRIGGER (Conditional) ---
    # This block simulates a UAV failure at a specific time if enabled by user.
    if SIM_FAILURE_TIME > 0 and simulation_time > SIM_FAILURE_TIME and simulation_time < SIM_FAILURE_TIME + TIME_STEP * 2:
        if SIM_ATTACK_STRATEGY == "Leader Direct Attack":
            # In Leader Direct Attack, if GBAD counter-attack is active, GBAD drops a follower engaging the target.
            engaging_followers = [u for u in uavs if
                                  not u.is_leader and u.is_active and u.status == UAVStatus.ENGAGING_TARGET and np.linalg.norm(
                                      u.pos_xy - GBAD_TARGET_POS_3D[:2]) < GBAD_EXCLUSION_RADIUS + 20]

            if engaging_followers:  # Randomly select one engaging follower to incapacitate.
                uav_to_incapacitate = random.choice(engaging_followers)
                # Mark UAV as inactive and incapacitated.
                uav_to_incapacitate.is_active = False
                uav_to_incapacitate.status = UAVStatus.INCAPACITATED
                uav_to_incapacitate.speed_xy = 0
                uav_to_incapacitate.vertical_speed = 0
                # No leader re-election needed here, as a follower is incapacitated.
            elif current_leader_obj:  # If no engaging followers, incapacitate the current leader.
                failed_leader_id = LEADER_ID
                elect_new_leader(uavs, failed_leader_id, selection_criteria="lowest_id")
        else:  # For other strategies, always incapacitate the current leader if applicable.
            if current_leader_obj:
                failed_leader_id = LEADER_ID
                elect_new_leader(uavs, failed_leader_id, selection_criteria="lowest_id")

        SIM_FAILURE_TIME = -2.0  # Set to a value that prevents this block from running again.
    # --- END UAV FAILURE TRIGGER ---

    simulation_time += TIME_STEP  # Advance simulation time.
    all_landed_or_incapacitated = True  # Flag to check if mission is complete.
    active_uav_count = 0  # Count active UAVs.

    # --- GBAD Lock-On Logic ---
    # Manages GBAD's targeting behavior based on time and UAV positions.
    if GBAD_lock_on_start_time > 0 and simulation_time >= GBAD_lock_on_start_time + GBAD_LOCK_ON_DURATION:
        GBAD_lock_on_start_time = -1.0  # Reset lock-on if duration is over.
        GBAD_target_uav_id = -1

    # If GBAD is not currently locked on, check for new targets.
    if GBAD_lock_on_start_time == -1.0:
        potential_targets_in_range = []
        for uav_obj in uavs:
            dist_to_GBAD = np.linalg.norm(uav_obj.pos_xy - GBAD_TARGET_POS_3D[:2])

            if uav_obj.is_active and dist_to_GBAD < GBAD_LOCK_ON_RANGE:
                # Only target UAVs in combat-relevant states.
                if uav_obj.status in [UAVStatus.ENGAGING_TARGET, UAVStatus.APPROACHING_TARGET,
                                      UAVStatus.EVADING, UAVStatus.POST_ENGAGE_CLEARING]:
                    potential_targets_in_range.append(uav_obj)

        # --- GBAD Target Acquisition Priority ---
        # Prioritize targets based on attack strategy.
        if SIM_ATTACK_STRATEGY == "Leader Direct Attack":
            # Prioritize incapacitating an actively engaging follower.
            follower_to_shoot_down = None
            for uav_obj in uavs:
                if not uav_obj.is_leader and uav_obj.is_active and uav_obj.status == UAVStatus.ENGAGING_TARGET and \
                        np.linalg.norm(uav_obj.pos_xy - GBAD_TARGET_POS_3D[:2]) < GBAD_EXCLUSION_RADIUS + 20:
                    follower_to_shoot_down = uav_obj
                    break  # Found one, GBAD will target this one.

            if follower_to_shoot_down:
                GBAD_target_uav_id = follower_to_shoot_down.id
                GBAD_lock_on_start_time = simulation_time  # Lock on immediately.

                # Incapacitate the selected follower UAV.
                follower_to_shoot_down.is_active = False
                follower_to_shoot_down.status = UAVStatus.INCAPACITATED
                follower_to_shoot_down.speed_xy = 0
                follower_to_shoot_down.vertical_speed = 0

            else:  # If no engaging follower, GBAD attempts to lock on other targets.
                # Prefer leader if in range.
                if current_leader_obj and current_leader_obj.is_leader and current_leader_obj.id in [u.id for u in
                                                                                                     potential_targets_in_range]:
                    GBAD_target_uav_id = current_leader_obj.id
                    GBAD_lock_on_start_time = simulation_time + GBAD_LOCK_ON_DELAY  # Apply delay for initial lock.
                elif potential_targets_in_range:  # Otherwise, pick a random active UAV.
                    random_target_uav = random.choice(potential_targets_in_range)
                    GBAD_target_uav_id = random_target_uav.id
                    GBAD_lock_on_start_time = simulation_time + GBAD_LOCK_ON_DELAY

        else:  # For Flanking Maneuver and Converge (or other strategies): GBAD picks a random active target.
            if current_leader_obj and current_leader_obj.is_leader and current_leader_obj.id in [u.id for u in
                                                                                                 potential_targets_in_range]:
                GBAD_target_uav_id = current_leader_obj.id
                GBAD_lock_on_start_time = simulation_time + GBAD_LOCK_ON_DELAY
            elif potential_targets_in_range:
                random_target_uav = random.choice(potential_targets_in_range)
                GBAD_target_uav_id = random_target_uav.id
                GBAD_lock_on_start_time = simulation_time + GBAD_LOCK_ON_DELAY

    # Update state for all UAVs.
    for uav_obj in uavs:
        if uav_obj.is_active and uav_obj.status != UAVStatus.LANDED:
            all_landed_or_incapacitated = False  # If any UAV is active and not landed, mission isn't complete.
        if uav_obj.is_active: active_uav_count += 1  # Count active UAVs.
        uav_obj.update_state(TIME_STEP)  # Call the UAV's update function.

    # --- Visualization Update ---
    for i, uav_obj in enumerate(uavs):
        if uav_obj.is_active:  # Draw active UAVs.
            uav_artists[i].set_data([uav_obj.pos_xy[0]], [uav_obj.pos_xy[1]])  # Update position.
            uav_artists[i].set_alpha(1.0)  # Full opacity.
            uav_altitude_texts[i].set_alpha(1.0)
            uav_path_lines[i].set_alpha(0.6)
            uav_direction_lines[i].set_alpha(0.8)

            # Update marker style for leader/followers.
            if uav_obj.id == LEADER_ID and uav_obj.is_leader:
                uav_artists[i].set_marker('^')
                uav_artists[i].set_markersize(10)
                uav_artists[i].set_color('red')
            else:
                uav_artists[i].set_marker('v')
                uav_artists[i].set_markersize(8)
                uav_artists[i].set_color('deepskyblue' if i % 2 == 1 else 'blue')

            # Update text display for UAV info.
            uav_altitude_texts[i].set_position((uav_obj.pos_xy[0], uav_obj.pos_xy[1] + UAV_RADIUS_VIS * 3.5))
            status_name_short = uav_obj.status.value

            is_leader_now = uav_obj.id == LEADER_ID and uav_obj.is_leader
            leader_text = " (L)" if is_leader_now else ""
            uav_altitude_texts[i].set_text(
                f"{uav_obj.id}{leader_text}: {uav_obj.alt:.0f}m H{math.degrees(uav_obj.hdg_rad):.0f} V{uav_obj.speed_xy:.1f}\n{status_name_short[:20]}")

            # Update path history line.
            if uav_obj.path_history_xy: px, py = zip(*uav_obj.path_history_xy); uav_path_lines[i].set_data(px, py)
            # Update direction line.
            line_len = uav_obj.speed_xy * 0.3 + 5 if uav_obj.status != UAVStatus.LANDED else 0
            dx, dy = math.cos(uav_obj.hdg_rad) * line_len, math.sin(uav_obj.hdg_rad) * line_len
            uav_direction_lines[i].set_data([uav_obj.pos_xy[0], uav_obj.pos_xy[0] + dx],
                                            [uav_obj.pos_xy[1], uav_obj.pos_xy[1] + dy])
        else:  # If UAV is inactive (incapacitated), fade it out.
            uav_artists[i].set_alpha(0.2)
            uav_altitude_texts[i].set_text(f"{uav_obj.id}\nINCAP")
            uav_altitude_texts[i].set_alpha(0.2)
            uav_path_lines[i].set_alpha(0.1)
            uav_direction_lines[i].set_alpha(0.1)

    # Update GBAD lock-on line visualization.
    if GBAD_lock_on_start_time > 0 and GBAD_target_uav_id != -1 and 0 <= GBAD_target_uav_id < len(uavs) and uavs[
        GBAD_target_uav_id].is_active:
        target_uav = uavs[GBAD_target_uav_id]
        GBAD_lock_on_line.set_data([GBAD_TARGET_POS_3D[0], target_uav.pos_xy[0]],
                                   [GBAD_TARGET_POS_3D[1], target_uav.pos_xy[1]])
    else:
        GBAD_lock_on_line.set_data([], [])  # Hide line if no lock-on.

    # Update the main title of the plot.
    title_text.set_text(
        f"Time: {simulation_time:.1f}s | Leader ({LEADER_ID if current_leader_obj else 'None'}) Status: {current_leader_obj.status.value if current_leader_obj else 'N/A'}")

    # --- Simulation Termination Conditions ---
    if all_landed_or_incapacitated or active_uav_count == 0:
        # Stop animation if all UAVs have landed or are incapacitated.
        if anim.event_source: anim.event_source.stop()
        GBAD_lock_on_line.set_data([], [])  # Hide lock-on line at end.

    if simulation_time >= MAX_SIM_TIME and not (all_landed_or_incapacitated or active_uav_count == 0):
        # Stop animation if max simulation time is reached.
        if anim.event_source: anim.event_source.stop()
        GBAD_lock_on_line.set_data([], [])  # Hide lock-on line at end.

    # Return all updated artists for efficient animation rendering.
    returned_artists = uav_artists + uav_altitude_texts + uav_path_lines + uav_direction_lines + \
                       [GBAD_marker, GBAD_exclusion_patch, base_marker, title_text, GBAD_lock_on_line] + nfz_patches
    return returned_artists


# --- Run Simulation ---
# Create the animation object, linking the figure, update function, and timing.
anim = animation.FuncAnimation(fig, update_simulation, frames=int(MAX_SIM_TIME / TIME_STEP),
                               interval=max(1, int(TIME_STEP * 150)), blit=False, repeat=False)
plt.show()  # Display the simulation animation.