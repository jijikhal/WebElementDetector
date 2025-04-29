import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def thousands_formatter(x, pos):
    if x == 0:
        return "0M"
    if x < 1000:
        return f'{x}'
    if x < 1_000_000:
        return f'{int(x/1000)}k'
    else:
        return f'{int(x/1000000)}M'
    
def exponential_moving_average(data, alpha=0.25):
    ema = [data[0]]
    for val in data[1:]:
        ema.append(alpha * val + (1 - alpha) * ema[-1])
    return np.array(ema)

data_cnn = np.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\rl\logs\v3CornerCNN\evaluations.npz")
data_bigger = np.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\rl\logs\v3BiggerNet\evaluations.npz")

mean_rewards_cnn = data_cnn["results"][:,0]
mean_rewards_cnn_smooth = exponential_moving_average(mean_rewards_cnn, alpha=0.1)
mean_rewards_bigger = data_bigger["results"][:,0]
mean_rewards_bigger_smooth = exponential_moving_average(mean_rewards_bigger, alpha=0.1)
timesteps = data_cnn["timesteps"]
timesteps2 = data_bigger["timesteps"]

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(timesteps, mean_rewards_cnn, color='red', alpha=0.3)
ax.plot(timesteps2, mean_rewards_bigger, color='blue', alpha=0.3)
ax.plot(timesteps, mean_rewards_cnn_smooth, label='CnnPolicy', color='red')
ax.plot(timesteps2, mean_rewards_bigger_smooth, label='BiggerNetPolicy', color='blue')
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

plt.xlabel('Time step')
plt.ylabel('Reward')
plt.title('Episode reward during training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()