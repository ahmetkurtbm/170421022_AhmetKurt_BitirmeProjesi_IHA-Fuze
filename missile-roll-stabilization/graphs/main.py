import pandas as pd
import matplotlib.pyplot as plt

filename = 'C:\\Projects\\Marmara\\Bitpro\\graphics\\data\\state_history_20250521-000752.xlsx'

for i in range(2, 8):
    df = pd.read_excel(filename, sheet_name=f'episode_{i}')

    # Set up time as row index (or use an actual time column if available)
    time = df.index  # Just using index as a proxy for time

    # Plot Roll vs Time
    plt.figure(figsize=(10, 4))
    plt.plot(time, df['roll'], color='blue', linewidth=1.5)
    plt.title('Roll vs Time')
    plt.xlabel('Time (index)')
    plt.ylabel('Roll (degrees)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Roll Rate vs Time
    # plt.figure(figsize=(10, 4))
    # plt.plot(time, df['roll_rate'], color='blue', linewidth=1.5)
    # plt.title('Roll Rate vs Time')
    # plt.xlabel('Time (index)')
    # plt.ylabel('Roll Rate (rad/s or deg/s)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()