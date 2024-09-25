import numpy as np
import random
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from src.agents.deep_q_agent import DeepQAgent
from src.data.chess_datamodule import ChessDataModule
from src.environements.chess_env import ChessEnv

# Initialize environment and agent
env = ChessEnv()
agent = DeepQAgent(number_of_actions=env.number_of_possible_moves)

# Set up training parameters
NUM_EPISODES = 1000
BATCH_SIZE = 32
NUM_WORKERS = 4
MAX_EPOCHS = 100

metrics: dict = {}
agent.fit(env=env)
