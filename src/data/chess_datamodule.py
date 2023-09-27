import chess
import gym
import numpy as np
import pytorch_lightning as pl
from gym import spaces
from torch.utils.data import DataLoader, Dataset


class ChessDataset(Dataset):
    def __init__(self, env, num_episodes):
        self.env = env
        self.num_episodes = num_episodes
        self.data = self.generate_episodes()

    def generate_episodes(self):
        episodes = []
        for _ in range(self.num_episodes):
            state = self.env.reset()
            done = False
            episode_data = []
            while not done:
                action = self.env.action_space.sample()  # Random action for now
                move = self.env._action_to_move(action)
                next_state, reward, done, _ = self.env.step(move)
                episode_data.append((state, action, reward, next_state, done))
                state = next_state
            episodes.append(episode_data)
        return episodes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ChessDataModule(pl.LightningDataModule):
    def __init__(self, env, num_episodes, batch_size, num_workers):
        super().__init__()
        self.env = env
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.chess_dataset = ChessDataset(self.env, self.num_episodes)

    def train_dataloader(self):
        return DataLoader(
            self.chess_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.chess_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.chess_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
