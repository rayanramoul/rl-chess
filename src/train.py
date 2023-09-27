import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from src.agents.deep_q_agent import DeepQAgent
from src.data.chess_datamodule import ChessDataModule
from src.environements.chess_env import ChessEnv

# Initialize environment and agent
env = ChessEnv()
agent = DeepQAgent(list_of_moves=env.list_of_moves)

# Set up training parameters
num_episodes = 1000
batch_size = 32
num_workers = 4
max_epochs = 100

datamodule = ChessDataModule(env, num_episodes, batch_size, num_workers)

# Set up PyTorch Lightning Trainer
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="chess_agent-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    monitor="val_loss",
    mode="min",
)

logger = TensorBoardLogger("logs", name="chess_agent")

trainer = pl.Trainer(
    max_epochs=max_epochs,
    callbacks=[checkpoint_callback, RichProgressBar()],
    logger=logger,
)

# Train the agent
trainer.fit(agent, datamodule=datamodule)
