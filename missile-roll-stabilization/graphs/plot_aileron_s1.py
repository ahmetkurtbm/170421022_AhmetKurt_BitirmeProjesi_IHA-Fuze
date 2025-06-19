import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Load the Excel file
excel_path = 'C:\\Projects\\Marmara\\Bitpro\\graphics\\data\\action_history_20250521-000752.xlsx'
sheet_names = [f'episode_{i}' for i in range(2, 8)]  # Sheets for episodes 2 to 7

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

# Plot each episode in its subplot
for idx, sheet in enumerate(sheet_names):
    df = pd.read_excel(excel_path, sheet_name=sheet)
    roll = df['saturated_aileron_pwm']
    time = df.index

    # Save aileron for combined plot later
    all_rolls.append(roll)

    color = cmap1(idx)
    # Plot in subplot
    axs[idx].plot(time, roll, color=color, linewidth=1)
    axs[idx].set_title(get_label(idx))
    axs[idx].set_xlabel('Time')
    axs[idx].set_ylabel('Aileron Input PWM (s)')
    axs[idx].grid(True)

    axs[idx].set_ylim(-0.21, 0.21)
    axs[idx].set_yticks([-0.2, -0.1, 0, 0.1, 0.2])  # Define positions

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
plt.show()

# Plot all rolls in a single figure
plt.figure(figsize=(16, 6))
for idx, roll in enumerate(all_rolls):
    plt.plot(roll.index, roll, color=cmap2(idx), label=get_label(idx), linewidth=1)

plt.title('Aileron PWM Comparisons')
plt.xlabel('Time')
plt.ylabel('Aileron Input PWM (s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
