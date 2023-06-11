import numpy as np
import random
import sys
sys.path.insert(1, '/Users/rayansamyramoul/Documents/Github/Arcane-Chess/chess_environement/chess_environement/')
from chess_environement.env import ChessEnv
from chess_agents.deep_q_agent import DeepQAgent

env = ChessEnv()
# Q-learning parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1
num_episodes = 1000

# Initialize Q-Network
agent = DeepQAgent() # network='conv',gamma=0.1,lr=0.07)
# R = Q_learning(agent,board)
len_episodes = 0

MAX_ITERATIONS_NUMBER = 100000

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    len_episodes += 1
    done = False
    for iteration_number in range(1, MAX_ITERATIONS_NUMBER): # while not done:
        
        # Epsilon-greedy exploration strategy
        if random.uniform(0, 1) < epsilon:
            action = agent.explore(env) # env.action_space.sample()  # Explore
        else:
            action = agent.exploit(env) # np.argmax(q_table[state])  # Exploit

        next_state, reward, done, _ = env.step(action)

        # Q-value update
        current_q = q_table[state, action]
        next_max_q = np.max(q_table[next_state])
        new_q = (1 - alpha) * current_q + alpha * (reward + gamma * next_max_q)
        q_table[state, action] = new_q

        state = next_state
        if done:
            break
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
