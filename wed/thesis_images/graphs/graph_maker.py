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

data_frozen = np.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\rl\logs\v4ResNetFrozen\evaluations.npz")
data_finetune = np.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\rl\logs\v4ResNetFineTune\evaluations.npz")
data_scratch = np.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\rl\logs\v4ResNetNotPretrained\evaluations.npz")
data_biggernet = np.load(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\rl\logs\v4BiggerNetFromScratch\evaluations.npz")

mean_rewards_frozen = data_frozen["results"][:500,0]
mean_rewards_frozen_smooth = exponential_moving_average(mean_rewards_frozen, alpha=0.1)
mean_rewards_pretrained = data_finetune["results"][:500,0]
mean_rewards_pretrained_smooth = exponential_moving_average(mean_rewards_pretrained, alpha=0.1)
mean_rewards_scratch = data_scratch["results"][:500,0]
mean_rewards_scratch_smooth = exponential_moving_average(mean_rewards_scratch, alpha=0.1)
mean_rewards_biggernet = data_biggernet["results"][:500,0]
mean_rewards_biggernet_smooth = exponential_moving_average(mean_rewards_biggernet, alpha=0.1)
timesteps = data_frozen["timesteps"][:500]


fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(timesteps, mean_rewards_frozen, color='red', alpha=0.3)
ax.plot(timesteps, mean_rewards_pretrained, color='blue', alpha=0.3)
ax.plot(timesteps, mean_rewards_scratch, color='green', alpha=0.3)
ax.plot(timesteps, mean_rewards_biggernet, color='yellow', alpha=0.3)
ax.plot(timesteps, mean_rewards_frozen_smooth, label='ResNet pre-trained frozen', color='red')
ax.plot(timesteps, mean_rewards_pretrained_smooth, label='ResNet pre-trained', color='blue')
ax.plot(timesteps, mean_rewards_scratch_smooth, label='ResNet not pre-trained', color='green')
ax.plot(timesteps, mean_rewards_biggernet_smooth, label='BiggerNetPolicy', color='yellow')
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

plt.xlabel('Time step')
plt.ylabel('Reward')
plt.title('Episode reward during training')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()