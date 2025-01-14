import gymnasium as gym
import numpy as np

alpha = 0.4 # Learning rate
gamma = 0.9 # Discount factor (how much rewards are valued)
epsilon = 0.7  # Exploration rate at start
episodes = 2000 # Number of training episodes

def train(env, qtable, episodes, alpha, gamma, epsilon):
    min_epsilon = 0.01
    epsilon_decay = 0.995 

    # Training loop
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        num_of_actions = 0

        while not done:
            if np.random.uniform(0,1) < epsilon:
                # Explore: Try a random action to discover new possibilities
                action = env.action_space.sample()
            else:
                # Exploit: Take the best-known action
                action = np.argmax(qtable[state])

            next_state, reward, done, truncated, info = env.step(action)
            num_of_actions += 1

            # Update Q-table
            qtable[state, action] = qtable[state, action] + alpha * (
                reward + gamma * np.max(qtable[next_state]) - qtable[state, action])

            state = next_state

            # Decay epsilon to reduce exploration over time
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
    print("Training completed.")

    return qtable

def evaluate(env, qtable):
    # Evaluation loop
    total_rewards = []
    total_steps = []
    eval_episodes = 10 
    max_steps = 200 # Maximum number of steps per episode

    for episode in range(eval_episodes):
        state = env.reset()[0]
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = np.argmax(qtable[state])  # Exploit in evaluation
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

        total_rewards.append(total_reward)
        total_steps.append(steps)

    # Compute and print the average total reward and average number of steps
    average_reward = np.mean(total_rewards)
    average_steps = np.mean(total_steps)

    print(f"Average total reward: {average_reward}")
    print(f"Average steps taken: {average_steps}")

    return

def main():
    # Create gymnasium environment
    env = gym.make("Taxi-v3",render_mode="ansi")
    env.reset()
    print(env.render())

    # Initialize Q-table
    action_size = env.action_space.n
    state_size = env.observation_space.n
    qtable = np.zeros([state_size, action_size])

    qtable = train(env, qtable, episodes, alpha, gamma, epsilon)
    evaluate(env, qtable)

if __name__ == "__main__":
    main()