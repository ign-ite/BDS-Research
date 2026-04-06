import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Assuming 'rewards' and 'returns' are your data arrays
window_size = 100
smoothed_rewards = moving_average(rewards, window_size)
smoothed_returns = moving_average(returns, window_size)

plt.plot(smoothed_rewards, label='Smoothed Average Reward')
plt.plot(smoothed_returns, label='Smoothed Average Return')
plt.legend()
plt.show()
