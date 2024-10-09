import gymnasium as gym
import numpy as np

alpha = 0.8 # Learning rate
gamma = 0.6 # Discount factor (how much rewards are valued)
epsilon = 0.9  # Exploration rate
episodes = 1000 # Number of training episodes

'''def manual():
    # Blue = passenger, purple = destination, yellow = taxi
    while not done:
        print(env.render())
        action = int(input('0/south 1/north 2/east 3/west 4/pickup 5/drop off: '))
        num_of_actions = num_of_actions + 1
        state, reward, done, truncated, info = env.step(action)
        time.sleep(1.0)
        print('')
        print(f'Observations: number of actions={num_of_actions}, reward={reward}, done={done}')'''

def train(env, qtable, episodes, alpha, gamma, epsilon):
    # Training loop
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        num_of_actions = 0
        min_epsilon = 0.01
        epsilon_decay = 0.995 

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

            # Epsilon decay: epsilon value is slowly reduced over time, to stable the results
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
            action = np.argmax(qtable[state])  # Always exploit in evaluation phase
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

        total_rewards.append(total_reward)
        total_steps.append(steps)

    # Compute and print the average total reward and average number of steps
    average_reward = np.mean(total_rewards)
    average_steps = np.mean(total_steps)

    print(f"Average Total Reward: {average_reward}")
    print(f"Average Steps Taken: {average_steps}")

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