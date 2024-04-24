import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load CSV file; change name to desired CSV file
df = pd.read_csv("outputs/basic_intersection_conn0_ep1.csv")

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
step = df.iloc[:, 0].values
y = df.iloc[:, 4].values


# Plot the graph
plt.plot(step, y,linestyle='-')
plt.xlabel('Step')
plt.ylabel('Speed')
plt.title('Plot of Step by System Mean Speed')
plt.grid(True)
plt.show()