import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def thousands_formatter(x, pos):
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

data_cnn = np.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\v2CNN\evaluations.npz")
data_mlp = np.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\ReinforcementLearning\logs\v2MLP\evaluations.npz")

mean_rewards_cnn = data_cnn["results"][:,0]
mean_rewards_cnn_smooth = exponential_moving_average(mean_rewards_cnn, alpha=0.25)
mean_rewards_mlp = data_mlp["results"][:,0]
mean_rewards_mlp_smooth = exponential_moving_average(data_mlp["results"][:,0], alpha=0.25)
timesteps = data_mlp["timesteps"]

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(timesteps, mean_rewards_cnn, color='red', alpha=0.3)
ax.plot(timesteps, mean_rewards_cnn_smooth, label='CnnPolicy', color='red')
ax.plot(timesteps, mean_rewards_mlp, color='blue', alpha=0.3)
ax.plot(timesteps, mean_rewards_mlp_smooth, label='MlpPolicy', color='blue')
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

plt.xlabel('Time step')
plt.ylabel('Reward')
plt.title('Episode reward during training')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()