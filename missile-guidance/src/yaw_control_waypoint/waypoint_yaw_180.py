# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 00:52:22 2025

@author: ammar
"""

from terminal_angle.yaw_control_waypoint.sr_env_yaw_180 import SRenv


# Step 1. create the gym environment
env = SRenv()

# Step 2. Run the simulation
env.run()

# Step 3. Save the flight data to Excel
env.save_data_to_csv()

# waypoint_save = env.get_initial_postition()

target_latitude: -5.234215686042022
target_longitude: -145.92349839296438

# In[] Bezier Curve

import numpy as np
import matplotlib.pyplot as plt

def bezier_curve(t, P0, P1, P2, P3):
    """Computes a point on a cubic Bézier curve for a given t."""
    x = (1 - t)**3 * P0[0] + 3 * (1 - t)**2 * t * P1[0] + 3 * (1 - t) * t**2 * P2[0] + t**3 * P3[0]
    y = (1 - t)**3 * P0[1] + 3 * (1 - t)**2 * t * P1[1] + 3 * (1 - t) * t**2 * P2[1] + t**3 * P3[1]
    return x, y

# Define control points
# P0 = (0, 0)
# P1 = (0, 2)
# P2 = (1, 1)
# P3 = (1, 3)
P0 = (-145.89275020309188, -5.131445862824855)
P3 = (-146.0000305900698, -5.257518959063409)
P1 = (P0[0], P0[1] + (P3[1] - P0[1])*2/3)
P2 = (P3[0], P3[1] - ((P3[1] - P0[1])*2/3))

# Generate curve points
t_values = np.linspace(0, 1, 8)  # 100 points along the curve
curve_points = np.array([bezier_curve(t, P0, P1, P2, P3) for t in t_values])

# Set figure size (width, height) with height 3 times the width
fig_width = 5  # You can adjust this value
fig_height = fig_width * 3
plt.figure(figsize=(fig_width, fig_height))

# Plot the Bézier curve
plt.plot(curve_points[:, 0], curve_points[:, 1], label="Missile Trajectory (Bézier Curve)", color="blue")

# Plot control points and lines
control_x = [P0[0], P1[0], P2[0], P3[0]]
control_y = [P0[1], P1[1], P2[1], P3[1]]
plt.scatter(control_x, control_y, color="red", label="Control Points")
plt.plot(control_x, control_y, '--', color="gray", label="Control Polygon")

plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Cubic Bézier Curve Trajectory")
plt.legend()
plt.grid()
plt.show()

# In[] Bezier Curve 2

import numpy as np
import matplotlib.pyplot as plt

def bezier_curve(t, P0, P1, P2, P3):
    """Computes a point on a cubic Bézier curve for a given t."""
    x = (1 - t)**3 * P0[0] + 3 * (1 - t)**2 * t * P1[0] + 3 * (1 - t) * t**2 * P2[0] + t**3 * P3[0]
    y = (1 - t)**3 * P0[1] + 3 * (1 - t)**2 * t * P1[1] + 3 * (1 - t) * t**2 * P2[1] + t**3 * P3[1]
    return x, y

# Define control points
P0 = (0, 0)
P1 = (0.0, 0.9)
P2 = (0.1, 1)
P3 = (1, 1)

# Generate curve points
t_values = np.linspace(0, 1, 12)  # 100 points along the curve
curve_points = np.array([bezier_curve(t, P0, P1, P2, P3) for t in t_values])

# Set figure size (width, height) with height 3 times the width
fig_width = 5  # You can adjust this value
fig_height = fig_width * 3
plt.figure(figsize=(fig_width, fig_height))

# Plot the Bézier curve
plt.plot(curve_points[:, 0], curve_points[:, 1], label="Missile Trajectory (Bézier Curve)", color="blue")

# Plot control points and lines
control_x = [P0[0], P1[0], P2[0], P3[0]]
control_y = [P0[1], P1[1], P2[1], P3[1]]
plt.scatter(control_x, control_y, color="red", label="Control Points")
plt.plot(control_x, control_y, '--', color="gray", label="Control Polygon")

plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Cubic Bézier Curve Trajectory")
plt.legend()
plt.grid()
plt.show()
