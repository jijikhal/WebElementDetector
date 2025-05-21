import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def thousands_formatter(x, pos):
    if x == 0:
        return "0"
    if x < 1000:
        return f'{x}'
    if x < 1_000_000:
        return f'{int(x/1000)}k'
    else:
        return f'{int(x/1000000)}M'

data = np.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\rl\logs\v0\evaluations.npz")

mean_rewards = data["results"][:,0]
timesteps = data["timesteps"]

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(timesteps, mean_rewards, label='Episode reward', color='red')
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

plt.xlabel('Time step')
plt.ylabel('Reward')
plt.title('Episode reward during training')
plt.grid(True)
plt.tight_layout()
plt.show()