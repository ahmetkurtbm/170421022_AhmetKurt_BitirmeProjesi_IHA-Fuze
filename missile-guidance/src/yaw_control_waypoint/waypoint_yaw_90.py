# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 18:09:46 2025

@author: ammar

currently working on "Launch Position 02"
"""

from terminal_angle.yaw_control_waypoint.sr_env_yaw_90 import SRenv


# Step 1. create the gym environment
env = SRenv()

# Step 2. Run the simulation
env.run()

# Step 3. Save the flight data to Excel
env.save_data_to_csv()

# waypoint_save = env.get_initial_postition()

