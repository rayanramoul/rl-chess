import numpy as np
import random
import sys
from chess_environement.env import ChessEnv
from chess_agents.deep_q_agent import DeepQAgent

env = ChessEnv()
# Q-learning parameters
alpha = 0.1
gamma = 0.6
epsilon_start = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
num_episodes = 1000
epsilon = epsilon_start
MAX_ITERATIONS_NUMBER = 100000

# Initialize Q-Network
agent = DeepQAgent() # network='conv',gamma=0.1,lr=0.07)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    for iteration_number in range(1, MAX_ITERATIONS_NUMBER): # while not done:
        
        # Epsilon-greedy exploration strategy
        if random.uniform(0, 1) < epsilon:
            action = agent.explore(env) # env.action_space.sample()  # Explore
        else:
            action = agent.exploit(env) # np.argmax(q_table[state])  # Exploit

        next_state, reward, done, _ = env.step(action)

        # Train the agent
        agent.train(state, action, reward, next_state, done)

        state = next_state
        if done:
            break
    
    # Decay epsilon
    epsilon = max(epsilon_min, epsilon_decay*epsilon)

# Evaluate the trained agent
num_eval_episodes = 10
total_rewards = 0

for _ in range(num_eval_episodes):
    state = env.reset()
    done = False

    while not done:
        action = agent.exploit(env)
        state, reward, done, _ = env.step(action)
        total_rewards += reward

average_reward = total_rewards / num_eval_episodes
print("Average reward:", average_reward)
