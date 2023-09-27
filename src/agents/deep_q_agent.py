import copy
import os
import pickle
import random
from collections import namedtuple

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# Constants
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
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


# DeepQAgent class converted to PyTorch Lightning module
class DeepQAgent(pl.LightningModule):
    def __init__(self, model=None, gamma=0.9, lr=0.001, list_of_moves=None):
        super(DeepQAgent, self).__init__()
        self.gamma = gamma
        self.number_of_actions = len(list_of_moves)
        self.list_of_moves = list_of_moves
        self.model = model if model else self.create_q_model()
        self.target_network = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)
        self.BATCH_SIZE = 128
        self.criterion = nn.MSELoss()
        self.model.to(self.device)

    def create_q_model(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=8 * 8, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.number_of_actions),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device).unsqueeze(0)
        return self.model(x)

    def predict(self, env):
        state_tensor = torch.tensor(env.translate_board()).unsqueeze(0)
        action_probs = self.model(state_tensor)
        move_number = torch.argmax(action_probs, dim=1).item()
        move = self.list_of_moves[move_number]
        return move, move_number

    def explore(self, env):
        move_number = random.randint(0, len(self.list_of_moves) - 1)
        move_str = self.list_of_moves[move_number]
        return move_str, move_number

    def exploit(self, env):
        state = env.translate_board()
        action_probs = self.forward(state)
        move_number = torch.argmax(action_probs[0], dim=0).item()
        move_str = self.list_of_moves[move_number]
        return move_str, move_number

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
        state, action, reward, next_state, done = batch
        self.memory.push(state, action, reward, next_state)
        self.optimize_model()
        if done:
            self.update_target_network()

    def configure_optimizers(self):
        return self.optimizer

    def save_pickle_agent(self, folder_path: str):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        torch.save(self.model.state_dict(), os.path.join(folder_path, "model.pth"))
        with open(os.path.join(folder_path, "deep_q_agent.pickle"), "wb") as f:
            pickle.dump(self, f)
