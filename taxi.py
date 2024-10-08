import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

alpha = 0.1 # Learning rate
gamma = 0.9 # Discount factor (how much rewards are valued)
epsilon = 0.2  # Exploration rate
episodes = 1000 # Number of training episodes
interactions = 10 

# Create gymnasium environment
env = gym.make("Taxi-v3",render_mode="ansi")
env.reset()
print(env.render())

# Initialize Q-table
action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros([state_size, action_size])

def manual():
    # Blue = passenger, purple = destination, yellow = taxi
    while not done:
        print(env.render())
        action = int(input('0/south 1/north 2/east 3/west 4/pickup 5/drop off: '))
        num_of_actions = num_of_actions + 1
        state, reward, done, truncated, info = env.step(action)
        time.sleep(1.0)
        print('')
        print(f'Observations: number of actions={num_of_actions}, reward={reward}, done={done}')


for episodes in range(episodes):
    state = env.reset()[0]
    done = False
    num_of_actions = 0
    total_rewards = 0

    while not done:
        if np.random.uniform(0,1) < epsilon:
            # Explore: Try a random action to discover new possibilities
            action = env.action_space.sample()
        else:
            # Exploit: Take the best-known action
             action = np.argmax(qtable[state])

        next_state, reward, done, truncated, info = env.step(action)
        num_of_actions = num_of_actions + 1

        # Q-learning equation
        qtable[state, action] = qtable[state, action] + alpha * (
            reward + gamma * np.max(qtable[next_state]) - qtable[state, action])

        
        state = next_state
    
print("Training completed.")