import numpy as np
import random
import sys
from chess_environement.env import ChessEnv
from chess_agents.deep_q_agent import DeepQAgent
import chess
import tqdm

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

NUMBER_OF_ACTIONS = len(env.list_of_moves)

# Initialize Q-Network
agent = DeepQAgent(list_of_moves=env.list_of_moves) # network='conv',gamma=0.1,lr=0.07)

# Training loop
for episode in tqdm.tqdm(range(num_episodes)):
    print("Episode: ", episode)
    state = env.reset()
    done = False
    episode_reward = 0
    for iteration_number in range(1, MAX_ITERATIONS_NUMBER): # while not done:
        
        # Epsilon-greedy exploration strategy
        if random.uniform(0, 1) < epsilon:
            move_str, move_number = agent.explore(env) # env.action_space.sample()  # Explore
        else:
            move_str, action = agent.exploit(env) # np.argmax(q_table[state])  # Exploit
        
        move_str = chess.Move.from_uci(move_str)
        # print("Move: ", move_str)
        next_state, reward, done, _ = env.step(move_str)

        # Train the agent
        agent.train(state, move_number, reward, next_state, done)

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