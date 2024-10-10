# Taxi-v3 Q-Learning Agent

This repository contains a Python implementation of a Q-learning agent to solve the **Taxi-v3** environment from the [Gymnasium library](https://gymnasium.farama.org/). The agent is trained using a Q-learning algorithm to pick up and drop off passengers efficiently in a grid-based environment. 

## Overview

**Taxi-v3** is a simple environment where a taxi must navigate a grid world to pick up a passenger at one location and drop them off at another. The agent receives rewards for completing the task and penalties for wrong actions or taking too many steps. The goal of this project is to train the taxi agent using the Q-learning algorithm to achieve an optimal policy, maximizing its rewards by minimizing the number of steps taken.

### Environment Details:
- **State space**: The grid consists of 500 discrete states (taxi position, passenger location, and destination).
- **Action space**: The agent can take 6 possible actions: move south, north, east, west, pick up the passenger, or drop them off.
- **Rewards**:
  - **+20** for a successful drop-off.
  - **-1** for each step taken.
  - **-10** for an incorrect pickup or drop-off.

## Q-Learning Algorithm

Q-learning is a reinforcement learning algorithm that learns the value of an action in a particular state by iteratively improving its estimate of the Q-values based on feedback from the environment.

### Hyperparameters:
- **Learning rate (`alpha`)**: 0.3 — Controls how much the agent updates its knowledge from new experiences.
- **Discount factor (`gamma`)**: 0.9 — Balances the importance of future rewards vs. immediate rewards.
- **Exploration rate (`epsilon`)**: 0.7 — Initial probability of taking random actions to explore the environment.
- **Epsilon decay (`epsilon_decay`)**: 0.999 — Gradually reduces exploration as training progresses.
- **Training episodes**: 2000 — The number of episodes the agent will play to learn the optimal policy.

## Project Structure
- `train()` : Implements the Q-learning algorithm to train the agent.
- `evaluate()` : Evaluates the learned policy after training, testing it for 10 episodes to compute the average reward and steps taken.
- `main()` : Orchestrates the training and evaluation process.

## Requirements

- Python 3.6+
- Gymnasium (formerly Gym)
- NumPy
