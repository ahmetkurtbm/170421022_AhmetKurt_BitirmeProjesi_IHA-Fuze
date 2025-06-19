import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Load the Excel file
excel_path = 'C:\\Projects\\Marmara\\Bitpro\\graphics\\data\\aileron_action_yaw180.xlsx'
sheet_name = 'Sheet1'

# Define color list for final plot
# colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange']
cmap1 = plt.get_cmap('coolwarm')
cmap2 = plt.get_cmap('tab10')

# Prepare to store roll data for final combined plot
all_rolls = []

# Create subplots 3x2
fig, axs = plt.subplots(3, 2, figsize=(16, 8))
fig.suptitle('Aileron PWM Over Time', fontsize=16)

# Flatten axs for easy indexing
axs = axs.flatten()

def get_label(idx):
    if idx == 0:
        return f'No Controller'
    if idx == 1:
        return f'PID'
    if idx == 2:
        return f'Linear Regression'
    if idx == 3:
        return f'LSTM'
    if idx == 4:
        return f'DDPG'
    if idx == 5:
        return f'PID + LR + LSTM + DDPG'

df = pd.read_excel(excel_path, sheet_name=sheet_name)
pid = df['PID']
lr = df['LR']
lstm = df['LSTM']
ddpg = df['DDPG']
ensemble = df['Ensemble']
roll_lists = [pid, lr, lstm, ddpg, ensemble]
time = df.index

# Find global axis limits
x_min, x_max = time.min(), 130
y_max = max([roll.max() for roll in roll_lists])
y_min = -y_max #_ min([roll.min() for roll in roll_lists])

# Optional: Add margin (padding)
y_margin = (y_max - y_min) * 0.1
y_min -= y_margin
y_max += y_margin

# Plot each episode in its subplot
for idx, df in enumerate(roll_lists):
    all_rolls.append(df)
    color = cmap1(idx)
    # Plot in subplot
    axs[idx].plot(time, df, color=color, linewidth=1)
    axs[idx].set_title(get_label(idx+1))
    axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Aileron Input PWM (s)')
    axs[idx].grid(True)

    # Apply common axis limits
    axs[idx].set_xlim(x_min, x_max)
    axs[idx].set_ylim(y_min, y_max)

# Remove the unused 6th subplot
fig.delaxes(axs[-1])
# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
plt.show()

# Plot all rolls in a single figure
plt.figure(figsize=(16, 6))
for idx, roll in enumerate(all_rolls):
    plt.plot(roll.index, roll, color=cmap2(idx), label=get_label(idx+1), linewidth=1)

plt.title('Aileron PWM Comparison')
plt.xlabel('Time')
plt.ylabel('Aileron Input PWM (s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
