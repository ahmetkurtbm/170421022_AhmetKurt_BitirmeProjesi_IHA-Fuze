# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 14:58:47 2024

@author: ammar
"""

from terminal_angle.pitch_terminal.sr_pitch_env_above80 import SRenv

# Step 1. create the gym environment
env = SRenv(terminal_pitch_target=-88)

# Step 2. Run the simulation
env.run()

# Step 3. Save the flight data to Excel
env.save_data_to_csv()

# waypoint_save = env.get_initial_postition()
