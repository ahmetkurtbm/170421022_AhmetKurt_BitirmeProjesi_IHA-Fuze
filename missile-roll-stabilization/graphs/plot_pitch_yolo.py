import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# open this to plot the pitch graph over time
# Load the Excel file
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# # File paths and sheet name
# excel_path1 = 'C:\\Projects\\Marmara\\Bitpro\\graphics\\data\\state_history_yolo_WITH_DISTURBANCE_NO_CONTROLLER4.xlsx'
# excel_path2 = 'C:\\Projects\\Marmara\\Bitpro\\graphics\\data\\state_history_yolo_YOLOv8N_NN.xlsx'
# sheet_name = 'episode_0'

# # Read Excel sheets
# df = pd.read_excel(excel_path1, sheet_name=sheet_name)
# df2 = pd.read_excel(excel_path2, sheet_name=sheet_name)

# # Ensure 'pitch' column exists
# if 'pitch' not in df.columns or 'pitch' not in df2.columns:
#     raise ValueError("Column 'pitch' not found in one of the sheets.")

# # Create a 1x2 subplot
# fig, axs = plt.subplots(1, 2, figsize=(16, 4), sharey=True)

# # First plot: with system fault
# axs[0].plot(df['pitch'], color=cm.coolwarm(0.0))
# axs[0].set_title('Pitch with System Fault')
# axs[0].set_xlabel('Time Step')
# axs[0].set_ylabel('Pitch (degrees)')
# axs[0].grid(True)
# # axs[0].set_ylim(-45, 45)

# # Second plot: with fault compensator
# axs[1].plot(df2['pitch'], color=cm.coolwarm(0.9))
# axs[1].set_title('Pitch with Fault Compensator')
# axs[1].set_xlabel('Time Step')
# axs[1].grid(True)
# # axs[1].set_ylim(-45, 45)

# # Improve layout
# plt.suptitle('Comparison of Pitch Over Time')
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
# plt.show()


# # aileron graph

# excel_path1 = 'C:\\Projects\\Marmara\\Bitpro\\graphics\\data\\action_history_yolo_WITH_DISTURBANCE_NO_CONTROLLER.xlsx'
# excel_path2 = 'C:\\Projects\\Marmara\\Bitpro\\graphics\\data\\action_history_yolo_YOLOv8N_NN.xlsx'
# sheet_name = 'episode_0'

# # Read the specific sheet
# df = pd.read_excel(excel_path1, sheet_name=sheet_name)
# df2 = pd.read_excel(excel_path2, sheet_name=sheet_name)

# # Ensure 'pitch' column exists
# if 'saturated_elevator_pwm' not in df.columns:
#     raise ValueError("Column 'saturated_elevator_pwm' not found in the sheet.")

# # Create a 1x2 subplot
# fig, axs = plt.subplots(1, 2, figsize=(16, 4), sharey=True)

# # First plot: with system fault
# axs[0].plot(df['saturated_elevator_pwm'], color=cm.coolwarm(0.0))
# axs[0].set_title('Aileron PWM with System Fault')
# axs[0].set_xlabel('Time Step')
# axs[0].set_ylabel('Pitch (degrees)')
# axs[0].grid(True)
# # axs[0].set_ylim(-45, 45)

# # Second plot: with fault compensator
# axs[1].plot(df2['saturated_elevator_pwm'], color=cm.coolwarm(0.9))
# axs[1].set_title('Aileron PWM with YOLOv8 as Fault Compensator')
# axs[1].set_xlabel('Time Step')
# axs[1].grid(True)
# # axs[1].set_ylim(-45, 45)

# # Improve layout
# plt.suptitle('Comparison of Aileron PWM')
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
# plt.show()


# for calculating the miss distance:
from math import sqrt, radians, cos, sin, atan2


# calculate the distance from lat long to kilometer
def calculate_distance(current_pos, target_pos, radius=1274.2, to_meter=True):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [current_pos[0], current_pos[1], target_pos[0], target_pos[1]])

    # Differences in latitude and longitude
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    # Haversine formula
    a = sin(delta_lat / 2)**2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance in the specified radius unit
    distance = radius * c
    return distance * 1000 if to_meter else distance

def classify_miss(distance):
    if distance <= 1:
        return 'Hit'
    elif distance <= 2.5:
        return 'Near Miss'
    else:
        return 'Miss'
    
# target_lat_long = [-5.23412004, -145.92274]
target_lat_long = [-5.234215694, -145.9234984]

# File paths and sheet name
excel_path1 = 'C:\\Projects\\Marmara\\Bitpro\\graphics\\data\\state_history_yolo_WITH_DISTURBANCE_NO_CONTROLLER4.xlsx'
excel_path2 = 'C:\\Projects\\Marmara\\Bitpro\\graphics\\data\\state_history_yolo_YOLOv8N_NN.xlsx'
sheet_name = 'episode_0'

# Read Excel sheets
df = pd.read_excel(excel_path1, sheet_name=sheet_name)
df2 = pd.read_excel(excel_path2, sheet_name=sheet_name)

print('no system fault miss:', calculate_distance([df['latitude'].iloc[59], df['longitude'].iloc[59]], target_lat_long) / 2.5)
print('YOLO miss:', calculate_distance([df2['latitude'].iloc[61], df2['longitude'].iloc[61]], target_lat_long) / 2.5)
