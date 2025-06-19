import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import sqrt, radians, cos, sin, atan2

# Load Excel file
file_path = 'C:\\Projects\\Marmara\\Bitpro\\graphics\\data\\position_yaw90.xlsx'
df = pd.read_excel(file_path)

# Controllers to plot
controllers = ['PID', 'LR', 'LSTM', 'DDPG', 'Ensemble']

# Target position
target_latitude = -5.23421606475547
target_longitude = -145.9234983903217

# Set up color map
cmap1 = plt.get_cmap('coolwarm')
cmap = plt.get_cmap('tab10')  # or 'viridis', 'plasma', 'Set1', etc.

# Calculate global axis limits
all_longs = []
all_lats = []
for ctrl in controllers:
    all_longs.append(df[f"{ctrl}_long"])
    all_lats.append(df[f"{ctrl}_lat"])

all_longs = pd.concat(all_longs)
all_lats = pd.concat(all_lats)

margin_x = 0.05 * (all_longs.max() - all_longs.min())
margin_y = 0.05 * (all_lats.max() - all_lats.min())

x_min, x_max = all_longs.min() - margin_x, all_longs.max() + margin_x
y_min, y_max = all_lats.min() - margin_y, all_lats.max() + margin_y

# Create 3x2 subplot (last one is combined)
fig, axs = plt.subplots(2, 3, figsize=(16, 10))
axs = axs.flatten()
fig.suptitle('Missile Trajectory per Controller and Combined', fontsize=16)


def get_max_index(idx):
    if idx == 0:
        return 119
    if idx == 1:
        return 145
    if idx == 2:
        return 122
    if idx == 3:
        return 125
    if idx == 4:
        return 126
    else:
        return -1

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


# Plot each controller in its own subplot
for idx, ctrl in enumerate(controllers):
    lat_col = f"{ctrl}_lat"
    long_col = f"{ctrl}_long"
    
    axs[idx].plot(df[long_col], df[lat_col], color=cmap1(idx), linewidth=1)
    
    # Mark initial position with 'o'
    axs[idx].plot(df[long_col].iloc[0], df[lat_col].iloc[0], 'o', color='green', markersize=8, label='Initial Position')
    # Mark target position with 'x'
    axs[idx].plot(target_longitude, target_latitude, 'x', color='red', markersize=10, markeredgewidth=2, label='Target Position')
    
    # Calculate miss distance (distance from last point to target at (0,0))
    final_lat = df[lat_col].iloc[get_max_index(idx)]
    final_long = df[long_col].iloc[get_max_index(idx)]

    miss_distance = calculate_distance([final_lat, final_long], [target_latitude, target_longitude])

    # miss_distance = sqrt(final_lat**2 + final_long**2)
    print(miss_distance)
    
    # Annotate miss distance
    axs[idx].text(0.05, 0.95, f"Miss Distance: {miss_distance:.2f}",
                  transform=axs[idx].transAxes,
                  verticalalignment='top',
                  fontsize=10,
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    


    axs[idx].set_title(f"{ctrl} Controller Trajectory")
    axs[idx].set_xlabel("Longitude")
    axs[idx].set_ylabel("Latitude")
    # axs[idx].grid(True)
    axs[idx].set_xlim(x_min, x_max)  # Set uniform x limits
    axs[idx].set_ylim(y_min, y_max)  # Set uniform y limits
    axs[idx].invert_yaxis()      # Keep if you want latitude axis inverted
    axs[idx].invert_xaxis()      # Flip longitude axis as requested
    axs[idx].set_aspect('equal', adjustable='box')
    axs[idx].legend()

    axs[idx].axhline(target_latitude, color='gray', linewidth=0.8, linestyle='--')  # Horizontal line at y = 0
    axs[idx].axvline(target_longitude, color='gray', linewidth=0.8, linestyle='--')  # Vertical line at x = 0

# Plot all in the last subplot
combined_ax = axs[-1]
for idx, ctrl in enumerate(controllers):
    lat_col = f"{ctrl}_lat"
    long_col = f"{ctrl}_long"
    combined_ax.plot(df[long_col], df[lat_col], label=ctrl, color=cmap(idx), linewidth=1)

# Mark initial positions for each controller in combined plot
for idx, ctrl in enumerate(controllers):
    lat_col = f"{ctrl}_lat"
    long_col = f"{ctrl}_long"
    combined_ax.plot(df[long_col].iloc[0], df[lat_col].iloc[0], 'o', color=cmap(idx), markersize=6)

# Mark target position on combined plot
combined_ax.plot(target_longitude, target_latitude, 'x', color='red', markersize=10, markeredgewidth=2, label='Target Position')

combined_ax.set_title("All Controller Trajectories")
combined_ax.set_xlabel("Longitude")
combined_ax.set_ylabel("Latitude")
# combined_ax.grid(True)
combined_ax.legend()
combined_ax.invert_yaxis()
combined_ax.invert_xaxis()
combined_ax.set_aspect('equal', adjustable='box')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
