import numpy as np
import random
import sys
sys.path.insert(1, '/Users/rayansamyramoul/Documents/Github/Arcane-Chess/chess_environement/chess_environement/')
from env import ChessEnv


env = ChessEnv()
# Q-learning parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1
num_episodes = 1000

# Initialize Q-table
num_states = 64
num_actions = 4096
q_table = np.zeros((num_states, num_actions))

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # Epsilon-greedy exploration strategy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        next_state, reward, done, _ = env.step(action)

        # Q-value update
        current_q = q_table[state, action]
        next_max_q = np.max(q_table[next_state])
        new_q = (1 - alpha) * current_q + alpha * (reward + gamma * next_max_q)
        q_table[state, action] = new_q

        state = next_state

# Evaluate the trained agent
num_eval_episodes = 10
total_rewards = 0

for _ in range(num_eval_episodes):
    state = env.reset()
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _ = env.step(action)
        total_rewards += reward

average_reward = total_rewards / num_eval_episodes
print("Average reward:", average_reward)
