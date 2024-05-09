import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

# Load CSV file; change name to desired CSV file

# col 0: step
# col 1: system_total_stopped
# col 2: system_total_waiting_time
# col 3: system_mean_waiting_time
# col 4: system_mean_speed
# col 5: t_stopped
# col 6: t_accumulated_waiting_time
# col 7: t_average_speed
# col 8: agents_total_stopped
# col 9: agents_total_accumulated_waiting_time

columns = {
    'system_total_stopped': 'Total Stopped',
    'system_total_waiting_time': 'Total Waiting Time',
    'system_mean_waiting_time': 'Mean Waiting Time',
    'agents_total_stopped': 'Agents Total Stopped',
    'agents_total_accumulated_waiting_time': 'Agents Total Waiting Time',
    'system_mean_speed': 'Mean Speed',
}

window_size = 51
poly_order = 1

def plot_stats(exp_name, rwd_fns):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axs = axs.flatten()

    for idx, (col, label) in enumerate(columns.items()):
        for rwd_fn in rwd_fns:
            output_file = 'outputs/' + exp_name + '_' + rwd_fn
            df_path = output_file + ".csv"
            df = pd.read_csv(df_path)

            # Apply Savitzky-Golay filter
            if len(df[col]) > window_size:  # Check if the data length is sufficient for the window size
                df['smoothed'] = savgol_filter(df[col], window_size, poly_order)
            else:
                df['smoothed'] = df[col]  # Not enough data, skip smoothing

            axs[idx].plot(df['step'], df['smoothed'], label=rwd_fn, linestyle='-', marker=None, linewidth=1.5)

        axs[idx].set_title(label)
        axs[idx].set_xlabel('Step')
        axs[idx].set_ylabel(label)
        axs[idx].grid(True)
        axs[idx].legend(title='Reward Function')

    plt.tight_layout()
    plt.savefig('outputs/' + exp_name + "_comparison.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    plot_stats('complex', ['lane-equal', 'improved-flow'])