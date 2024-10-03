import copy
import os
import pickle
import random
from collections import namedtuple
from rich.progress import track

# import overrides
from typing import List, override

import numpy as np
from loguru import logger
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import wandb

# Constants
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done", "player")
)
CHESS_DICT = {
    "p": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "P": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "n": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "N": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "b": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "B": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "r": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "R": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "q": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "Q": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "k": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "K": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ".": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()


# Utility functions
def translate_board(board):
    pgn = board.epd()
    foo = []
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        foo2 = []
        for thing in row:
            if thing.isdigit():
                foo2.extend([CHESS_DICT["."]] * int(thing))
            else:
                foo2.append(CHESS_DICT[thing])
        foo.append(foo2)
    return np.array(foo)


def board_matrix(board):
    pgn = board.epd()
    foo = []
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        for thing in row:
            if thing.isdigit():
                foo.extend(["."] * int(thing))
            else:
                foo.append(thing)
    return np.array(foo)


def translate_move(move):
    return np.array([move.from_square, move.to_square])


# Replay Memory class
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepQNetwork(nn.Module):
    def __init__(self, input_channels, action_space):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Calculate the size of the flattened output from conv layers
        self.fc_input_dim = 64 * 8 * 8

        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, self.fc_input_dim)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class DeepQAgent(torch.nn.Module):
    def __init__(
        self,
        model=None,
        gamma=0.9,
        lr=0.001,
        batch_size=8,
        number_of_actions: int = 8,
        number_episodes: int = 10,
        target_update: int = 1000,
        replay_memory_size: int = 10000,
        optimize_every_n_steps: int = 10,
    ) -> None:
        super(DeepQAgent, self).__init__()
        self.gamma = gamma
        self.number_of_actions = number_of_actions
        self.model = None
        self.memory = ReplayMemory(10000)
        self.BATCH_SIZE = batch_size
        self.NUMBER_EPISODES = number_episodes
        self.REPLAY_MEMORY_SIZE = replay_memory_size
        self.OPTIMIZE_EVERY_N_STEPS = optimize_every_n_steps
        self.LEARNING_RATE = lr
        self.TARGET_UPDATE = target_update
        self.config = {
            "gamma": self.gamma,
            "lr": self.LEARNING_RATE,
            "batch_size": self.BATCH_SIZE,
            "number_of_actions": self.number_of_actions,
            "number_episodes": self.NUMBER_EPISODES,
            "replay_memory_size": self.REPLAY_MEMORY_SIZE,
            "optimize_every_n_steps": self.OPTIMIZE_EVERY_N_STEPS,
        }
        assert self.BATCH_SIZE <= self.REPLAY_MEMORY_SIZE
        assert self.BATCH_SIZE <= self.OPTIMIZE_EVERY_N_STEPS
        self.criterion = nn.MSELoss()
        self.device = "cpu"

    def create_q_model(self, action_space: int) -> None:
        self.model = DeepQNetwork(1, action_space=action_space)
        self.model.to(self.device)
        self.target_network = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

    def forward(self, x):
        return self.model(x)

    def predict(self, state, env):
        state_tensor = (
            torch.tensor(state, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            action_probs = self.model(state_tensor)

        # Filter out illegal moves
        legal_moves = env.available_moves
        legal_move_indices = [env.move_str_to_index(str(move)) for move in legal_moves]
        legal_action_probs = action_probs[0, legal_move_indices]

        move_index = legal_move_indices[torch.argmax(legal_action_probs).item()]
        move = env.move_index_to_str(move_index)
        return move, move_index

    def explore(self, env):
        return random.choice(env.available_moves)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        ).to(self.device)

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        player_batch = torch.tensor(batch.player, device=self.device)

        state_batch = (
            torch.cat(batch.state).to(self.device).unsqueeze(1)
        )  # Adding channel dimension
        model_output = self.model(state_batch)
        state_action_values = model_output.gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            # swap dim 0 and dim 1 of non_final_next_states
            non_final_next_states = non_final_next_states.unsqueeze(1)
            next_state_values[non_final_mask] = self.target_network(
                non_final_next_states
            ).max(1)[0]

        # Adjust rewards based on the player
        adjusted_rewards = reward_batch * (2 * player_batch - 1)
        expected_state_action_values = (
            next_state_values * self.gamma
        ) + adjusted_rewards

        loss = self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.model.state_dict())

    def save_agent(self, folder_path: str):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        torch.save(self.model.state_dict(), os.path.join(folder_path, "model.pth"))
        torch.save(
            self.target_network.state_dict(),
            os.path.join(folder_path, "target_network.pth"),
        )
        with open(os.path.join(folder_path, "deep_q_agent.pickle"), "wb") as f:
            pickle.dump(self, f)

    def load_agent(self, folder_path: str):
        self.model.load_state_dict(torch.load(os.path.join(folder_path, "model.pth")))
        self.target_network.load_state_dict(
            torch.load(os.path.join(folder_path, "target_network.pth"))
        )
        self.model.to(self.device)
        self.target_network.to(self.device)

    def fit(self, env: gym.Env) -> None:
        wandb.init(project="chess-rl", config=self.config)
        epsilon_start = 1.0
        epsilon_end = 0.1
        epsilon_decay = 0.995
        epsilon = epsilon_start
        self.create_q_model(action_space=env.action_space)

        for episode_idx in track(
            range(self.NUMBER_EPISODES), description="Training Episode"
        ):
            state = env.reset()
            done = False
            total_reward = 0
            moves = 0

            while not done:
                player = moves % 2  # 0 for white, 1 for black

                if random.random() < epsilon:
                    move = self.explore(env)
                    move_idx = env.move_str_to_index(str(move))
                else:
                    move, move_idx = self.predict(state, env)

                next_state, reward, done = env.step(move)

                self.memory.push(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                    move_idx,
                    reward,
                    torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                    if not done
                    else None,
                    done,
                    player,
                )

                state = next_state
                total_reward += reward
                moves += 1

                if moves % self.OPTIMIZE_EVERY_N_STEPS == 0:
                    loss = self.optimize_model()
                    wandb.log({"loss": loss})

                if moves % self.TARGET_UPDATE == 0:
                    self.update_target_network()

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            wandb.log(
                {
                    "episode": episode_idx,
                    "total_reward": total_reward,
                    "moves": moves,
                    "epsilon": epsilon,
                }
            )

            if episode_idx % 100 == 0:
                self.save_agent(f"checkpoints/episode_{episode_idx}")

        wandb.finish()
