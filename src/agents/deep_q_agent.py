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

# Constants
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
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


class DeepQAgent(torch.nn.Module):
    def __init__(
        self,
        model=None,
        gamma=0.9,
        lr=0.001,
        batch_size=8,
        number_of_actions: int = 8,
        number_episodes: int = 10,
        replay_memory_size: int = 10000,
        optimize_every_n_steps: int = 10,
    ) -> None:
        super(DeepQAgent, self).__init__()
        self.gamma = gamma
        self.number_of_actions = number_of_actions
        self.model = model if model else self.create_q_model()
        self.target_network = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)
        self.BATCH_SIZE = batch_size
        self.NUMBER_EPISODES = number_episodes
        self.REPLAY_MEMORY_SIZE = replay_memory_size
        self.OPTIMIZE_EVERY_N_STEPS = optimize_every_n_steps
        assert self.BATCH_SIZE <= self.REPLAY_MEMORY_SIZE
        assert self.BATCH_SIZE <= self.OPTIMIZE_EVERY_N_STEPS
        self.criterion = nn.MSELoss()
        self.device = "cpu"
        self.model.to(self.device)

    def create_q_model(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8 * 8, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.number_of_actions),
            nn.Softmax(dim=1),
        )

    def forward(self, x, env=None):
        x = torch.tensor(x, dtype=torch.float).to(self.device).unsqueeze(0)
        y = self.model(x)
        # if available_moves is not None, we argmax on the available moves
        if env:
            available_moves = env.available_moves
            dict_moves = env.all_possibles_moves_list
            index_feasible_moves = [dict_moves[str(move)] for move in available_moves]
            index_non_feasible_moves = [
                i
                for i in range(self.number_of_actions)
                if i not in index_feasible_moves
            ]
            # set the non feasible indexes to zero in the output
            y[:, index_non_feasible_moves] = 0

        # return argmax of the available
        return y

    def predict(self, env):
        state_tensor = torch.tensor(env.translate_board()).unsqueeze(0)
        action_probs = self.model(state_tensor)
        move_number = torch.argmax(action_probs, dim=1).item()
        move = self.list_of_moves[move_number]
        return move, move_number

    def explore(self, env):
        return random.choice(env.available_moves)

    def exploit(self, env):
        state = env.translate_board()
        action_probs = self.forward(state, env)
        move_idx = torch.argmax(action_probs[0], dim=0).item()
        move_str = env.move_index_to_str(move_idx)
        return move_str, move_idx

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool
        )
        next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = (
            torch.cat(next_states).view(-1, 8, 8).unsqueeze(1).to(self.device)
        )

        state_batch = torch.cat(batch.state).view(-1, 8, 8).unsqueeze(1).to(self.device)
        action_batch = torch.tensor(batch.action).view(-1, 1).to(self.device)
        reward_batch = torch.tensor(batch.reward).view(-1, 1).to(self.device)

        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = (
            self.target_network(non_final_next_states).max(1)[0].detach()
        )
        expected_state_action_values = (
            next_state_values * self.gamma
        ) + reward_batch.squeeze()

        loss = self.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.model.state_dict())

    def training_step(self, batch, batch_idx):
        self.optimize_model()

    def configure_optimizers(self):
        return self.optimizer

    def save_pickle_agent(self, folder_path: str):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        torch.save(self.model.state_dict(), os.path.join(folder_path, "model.pth"))
        with open(os.path.join(folder_path, "deep_q_agent.pickle"), "wb") as f:
            pickle.dump(self, f)

    def fit(self, env: gym.Env) -> None:
        replay_memory = ReplayMemory(self.REPLAY_MEMORY_SIZE)
        metrics = {"episode": [], "loss_episode": [], "average_reward_episode": []}
        for episode_idx in track(
            range(self.NUMBER_EPISODES), description="Training Episode"
        ):
            finished_game = False
            self.board = env.reset()
            while not finished_game:
                epsilon = random.uniform(0, 1)
                if epsilon < self.gamma:
                    move_str = self.explore(env)
                    move_idx = env.move_str_to_index(move_str)
                else:
                    move_str, move_idx = self.exploit(env)

                state = env.translate_board()

                # do the action in the environment
                next_state, reward, done = env.step(move_str)

                # print differences between the 2 matrices
                # assert (state != next_state).any(), "No action was made"

                # store the transition in the replay memory
                replay_memory.push(state, move_idx, reward, next_state, done)

                # optimize the model
                if episode_idx > 1 and episode_idx % self.OPTIMIZE_EVERY_N_STEPS == 0:
                    self.training_step(
                        replay_memory.sample(self.BATCH_SIZE), episode_idx
                    )
                finished_game = done
