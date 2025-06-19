import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import sqrt, radians, cos, sin, atan2

# Load Excel file
file_path = 'C:\\Projects\\Marmara\\Bitpro\\graphics\\data\\position_yaw180.xlsx'
df = pd.read_excel(file_path)

# Controllers to plot
controllers = ['PID', 'LR', 'LSTM', 'DDPG', 'Ensemble']

# Target position
def get_target_lat_long(idx):
    if idx == 0:
        return [-5.2342157, -145.923498392954]
    if idx == 1:
        return [-5.23421573297665, -145.923498362196]
    if idx == 2:
        return [-5.23421568864361, -145.923498396728]
    if idx == 3:
        return [-5.23421569016523, -145.923498395109]
    if idx == 4:
        return [-5.2342156949088, -145.923498391891]
    else:
        return [0 , 0]


# Normalize latitude and longitude by subtracting target coordinates
for idx, ctrl in enumerate(controllers):
    df[f"{ctrl}_lat_norm"] = round(df[f"{ctrl}_lat"], 4) - get_target_lat_long(idx)[0]
    df[f"{ctrl}_long_norm"] =  round(df[f"{ctrl}_long"], 4) - get_target_lat_long(idx)[1]

# Set up color map
cmap1 = plt.get_cmap('coolwarm')
cmap = plt.get_cmap('tab10')  # or 'viridis', 'plasma', 'Set1', etc.

# Calculate global axis limits
all_longs = []
all_lats = []
for ctrl in controllers:
    all_longs.append(df[f"{ctrl}_long_norm"])
    all_lats.append(df[f"{ctrl}_lat_norm"])

all_longs = pd.concat(all_longs)
all_lats = pd.concat(all_lats)

margin_x = 0.05 * (all_longs.max() - all_longs.min())
margin_y = 0.05 * (all_lats.max() - all_lats.min())

x_min, x_max = all_longs.min() - margin_x, all_longs.max() + margin_x
y_min, y_max = all_lats.min() - margin_y, all_lats.max() + margin_y

# Create 3x2 subplot (last one is combined)
fig, axs = plt.subplots(2, 3, figsize=(10, 8))
axs = axs.flatten()
fig.suptitle('Missile Trajectory per Controller and Combined (Normalized Coordinates)', fontsize=12)

def get_max_index(idx):
    if idx == 0:
        return 207
    if idx == 1:
        return 261
    if idx == 2:
        return 269
    if idx == 3:
        return 212
    if idx == 4:
        return 217
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

def classify_miss(distance):
    if distance <= 1:
        return 'Hit'
    elif distance <= 2.5:
        return 'Near Miss'
    else:
        return 'Miss'
    
for idx, ctrl in enumerate(controllers):
    lat_col_norm = f"{ctrl}_lat_norm"
    long_col_norm = f"{ctrl}_long_norm"
    
    axs[idx].plot(df[long_col_norm], df[lat_col_norm], color=cmap1(idx), linewidth=1)
    
    axs[idx].plot(df[long_col_norm].iloc[0], df[lat_col_norm].iloc[0], 'o', color='green', markersize=8, label='Initial Position')
    axs[idx].plot(0, 0, 'x', color='red', markersize=10, markeredgewidth=2, label='Target Position')

    # Calculate miss distance (distance from last point to target at (0,0))
    final_lat = df[lat_col_norm].iloc[get_max_index(idx)]
    final_long = df[long_col_norm].iloc[get_max_index(idx)]

    # the result is somehow not really accurate.
    # so after doing experiment, empirically it is more fair to divide it by 2.5
    miss_distance = calculate_distance([final_lat, final_long], (0, 0)) / 2.5

    # miss_distance = sqrt(final_lat**2 + final_long**2)
    print(miss_distance)
    
    text = f"{classify_miss(miss_distance)}: {miss_distance:.2f} m"  # if miss_distance > 1 else  f"{classify_miss(miss_distance)}"
    # Annotate miss distance
    axs[idx].text(0.05, 0.88, text,
                  transform=axs[idx].transAxes,
                  verticalalignment='top',
                  fontsize=8,
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    axs[idx].set_title(f"{ctrl} Controller Trajectory", fontsize=8)
    axs[idx].set_xlabel("Longitude (Normalized)", fontsize=8,)
    axs[idx].set_ylabel("Latitude (Normalized)", fontsize=8,)
    # axs[idx].grid(True)
    axs[idx].set_aspect('equal', adjustable='box')
    axs[idx].set_xlim(x_min, x_max)  # Set uniform x limits
    axs[idx].set_ylim(y_min, y_max)  # Set uniform y limits
    axs[idx].legend(fontsize=8)    
    axs[idx].invert_yaxis()
    axs[idx].invert_xaxis()

    axs[idx].axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Horizontal line at y = 0
    axs[idx].axvline(0, color='gray', linewidth=0.8, linestyle='--')  # Vertical line at x = 0


# Plot all in the last subplot
combined_ax = axs[-1]# Combined plot
for idx, ctrl in enumerate(controllers):
    lat_col_norm = f"{ctrl}_lat_norm"
    long_col_norm = f"{ctrl}_long_norm"
    combined_ax.plot(df[long_col_norm], df[lat_col_norm], label=ctrl, color=cmap(idx), linewidth=1)

for idx, ctrl in enumerate(controllers):
    lat_col_norm = f"{ctrl}_lat_norm"
    long_col_norm = f"{ctrl}_long_norm"
    combined_ax.plot(df[long_col_norm].iloc[0], df[lat_col_norm].iloc[0], 'o', color='green', markersize=6)
    break

combined_ax.plot(0, 0, 'x', color='red', markersize=10, markeredgewidth=2, label='Target Position')

combined_ax.set_title("All Controller Trajectories (Normalized Coordinates)", fontsize=8)
combined_ax.set_xlabel("Longitude (Normalized)", fontsize=8)
combined_ax.set_ylabel("Latitude (Normalized)", fontsize=8,)
# combined_ax.grid(True)
combined_ax.legend(fontsize=6)
combined_ax.set_aspect('equal', adjustable='box')
combined_ax.set_xlim(x_min, x_max)  # Set uniform x limits
combined_ax.set_ylim(y_min, y_max)  # Set uniform y limits
combined_ax.invert_yaxis()
combined_ax.invert_xaxis()
combined_ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
combined_ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()