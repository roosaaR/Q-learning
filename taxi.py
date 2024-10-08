import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt


alpha = 0.1 # learning rate

# Create gymnasium environment
env = gym.make("Taxi-v3",render_mode="ansi")
env.reset()
print(env.render())
