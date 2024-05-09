from run import runExperiment
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import griddata
from matplotlib import cm

from pathlib import Path

result_path = 'outputs/CV_results'

def CV(exp_name = '2lane', rwd_fn = 'avg-speed'):
    # Define ranges for alpha, gamma, and decay
    alphas = np.linspace(0.1, 0.9, 5)
    gammas = np.linspace(0.96, 0.98, 6)
    decays = np.linspace(0.997, 0.9999, 5)

    results = []

    # Cross-validation
    best_params = {}
    best_score = -float('inf')

    for a in alphas:
        for g in gammas:
            for d in decays:
                avg_reward = runExperiment(n_sec = 3000, gui = False, n_runs = 1, # 3-fold CV
                                           exp_name=exp_name, rwd_fns=[rwd_fn],
                                        alpha = a, gamma = g, decay = d)[rwd_fn]
                
                results.append((a, g, d, avg_reward))

                if avg_reward > best_score:
                    best_score = avg_reward
                    best_params = {'alpha': a, 'gamma': g, 'decay': d}

    print("Best parameters:", best_params)

    df = pd.DataFrame(results, columns=['alpha', 'gamma', 'decay', 'reward'])

    Path(Path(result_path).parent).mkdir(parents=True, exist_ok=True)
    df.to_csv(result_path+".csv", index=False)

# CV(exp_name = '2lane', rwd_fn = 'avg-speed')

df = pd.read_csv(result_path+".csv")

# Create grid for interpolation
grid_x, grid_y = np.mgrid[min(df['alpha']):max(df['alpha']):100j, min(df['gamma']):max(df['gamma']):100j]

# Interpolate decay and reward
grid_decay = griddata((df['alpha'], df['gamma']), df['decay'], (grid_x, grid_y), method='cubic')
grid_reward = griddata((df['alpha'], df['gamma']), df['reward'], (grid_x, grid_y), method='cubic')

print(np.min(df['reward']), np.max(df['reward']))
# Normalize reward for coloring
norm = plt.Normalize(vmin=np.min(df['reward']), vmax=np.max(df['reward']))
colors = cm.viridis(norm(grid_reward))

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(grid_x, grid_y, grid_decay, facecolors=colors, rstride=3, cstride=3, antialiased=False, shade=False)

ax.set_xlabel('Alpha')
ax.set_ylabel('Gamma')
ax.set_zlabel('Decay')
ax.set_title('Surface Plot of Decay over Alpha and Gamma with Reward Coloring')

# Color bar for the rewards
m = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
m.set_array([])
cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Reward')

plt.savefig('outputs/CV_results.png', dpi=300)
plt.show()