import copy
import os
import pickle
import random
from collections import namedtuple, deque
from rich.progress import (
    track,
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich import print as rprint

# import overrides
from typing import List, override

import numpy as np
from loguru import logger
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import wandb

# Constants
Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done", "player", "priority"),
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
console = Console()


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


# Prioritized Replay Memory class
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.memory = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        self.max_priority = 1.0

    def push(self, *args):
        """Saves a transition with priority."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(self.max_priority)

        # Add priority to the transition
        transition_with_priority = Transition(*args, priority=self.max_priority)
        self.memory[self.position] = transition_with_priority
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None, None, None

        # Calculate sampling probabilities
        priorities = np.array(list(self.priorities)[: len(self.memory)])
        probs = priorities**self.alpha
        probs /= probs.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Get transitions
        transitions = [self.memory[idx] for idx in indices]

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return transitions, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.memory)


# Backward compatibility: alias for old checkpoints
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # Handle both old and new transition formats
        if len(args) == 6:  # Old format without priority
            self.memory[self.position] = Transition(*args, priority=1.0)
        else:  # New format with priority
            self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), None, None

    def __len__(self):
        return len(self.memory)


class DeepQNetwork(nn.Module):
    def __init__(self, input_channels, action_space):
        super(DeepQNetwork, self).__init__()
        # Input is 8x8x12 (board representation)

        # Convolutional layers with residual connections
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Residual blocks
        self.res_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.res_bn1 = nn.BatchNorm2d(256)
        self.res_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.res_bn2 = nn.BatchNorm2d(256)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, batch_first=True
        )

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_input_dim = 256 + 1  # +1 for the turn information

        # Fully connected layers with better architecture
        self.fc1 = nn.Linear(self.fc_input_dim, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(256, action_space)

        # Value and advantage streams for Dueling DQN
        self.value_stream = nn.Linear(256, 1)
        self.advantage_stream = nn.Linear(256, action_space)

    def forward(self, x, turn):
        # x shape: (batch_size, 8, 8, 12) -> need to transpose to (batch_size, 12, 8, 8)
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)  # (batch_size, 12, 8, 8)

        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Residual block
        residual = x
        x = F.relu(self.res_bn1(self.res_conv1(x)))
        x = self.res_bn2(self.res_conv2(x))
        x = F.relu(x + residual)  # Skip connection

        # Attention mechanism (reshape for attention)
        batch_size, channels, height, width = x.shape
        x_flat = x.view(batch_size, channels, -1).permute(
            0, 2, 1
        )  # (batch_size, 64, 256)
        x_attended, _ = self.attention(x_flat, x_flat, x_flat)
        x = x_attended.permute(0, 2, 1).view(batch_size, channels, height, width)

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Concatenate with turn information
        x = torch.cat([x, turn.float().unsqueeze(1)], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        # Dueling DQN: separate value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class DeepQAgent(torch.nn.Module):
    def __init__(
        self,
        model=None,
        gamma=0.99,
        lr=0.0001,
        batch_size=64,
        number_of_actions: int = 8,
        number_episodes: int = 10,
        epsilon_decay: float = 0.9995,
        target_update: int = 500,
        replay_memory_size: int = 50000,
        optimize_every_n_steps: int = 4,
        continue_from_checkpoint_path: str = "",
        save_every_n_episodes: int = 100,
    ) -> None:
        super(DeepQAgent, self).__init__()
        self.gamma = gamma
        self.number_of_actions = number_of_actions
        self.model = None
        self.memory = PrioritizedReplayMemory(replay_memory_size)
        self.BATCH_SIZE = batch_size
        self.NUMBER_EPISODES = number_episodes
        self.REPLAY_MEMORY_SIZE = replay_memory_size
        self.OPTIMIZE_EVERY_N_STEPS = optimize_every_n_steps
        self.LEARNING_RATE = lr
        self.TARGET_UPDATE = target_update
        self.EPSILON_DECAY = epsilon_decay
        self.continue_from_checkpoint_path = continue_from_checkpoint_path
        self.save_every_n_episodes = save_every_n_episodes

        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []
        self.win_rate = 0.0
        self.avg_game_length = 0.0

        self.config = {
            "gamma": self.gamma,
            "lr": self.LEARNING_RATE,
            "batch_size": self.BATCH_SIZE,
            "number_of_actions": self.number_of_actions,
            "number_episodes": self.NUMBER_EPISODES,
            "replay_memory_size": self.REPLAY_MEMORY_SIZE,
            "optimize_every_n_steps": self.OPTIMIZE_EVERY_N_STEPS,
            "target_update": self.TARGET_UPDATE,
            "epsilon_decay": self.EPSILON_DECAY,
            "continue_from_checkpoint_path": self.continue_from_checkpoint_path,
            "save_every_n_episodes": self.save_every_n_episodes,
        }
        assert self.BATCH_SIZE <= self.REPLAY_MEMORY_SIZE
        # Use Huber loss for better stability
        self.criterion = nn.SmoothL1Loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rprint(f"[bold green]Using device: {self.device}[/bold green]")

    def create_q_model(self, action_space: int) -> None:
        # Chess board has 12 channels (6 piece types x 2 colors)
        self.model = DeepQNetwork(12, action_space=action_space)
        self.model.to(self.device)
        self.target_network = copy.deepcopy(self.model)
        # Use AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.LEARNING_RATE, weight_decay=1e-4
        )
        # Use cosine annealing scheduler for better convergence
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.NUMBER_EPISODES, eta_min=1e-6
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, state, env, turn):
        if isinstance(turn, int):
            turn = torch.tensor([turn], dtype=torch.float32).to(self.device)

        # State is already in the correct format (8, 8, 12)
        state_tensor = (
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            action_probs = self.model(state_tensor, turn=turn)

        # Filter out illegal moves
        legal_moves = env.available_moves
        legal_move_indices = [env.move_str_to_index(str(move)) for move in legal_moves]

        if not legal_move_indices:
            # No legal moves available, return a default move
            return "a1a1", 0

        legal_action_probs = action_probs[0, legal_move_indices]

        move_index = legal_move_indices[torch.argmax(legal_action_probs).item()]
        move = env.move_index_to_str(move_index)
        return move, move_index

    def explore(self, env):
        return random.choice(env.available_moves)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return None

        sample_result = self.memory.sample(self.BATCH_SIZE)
        if sample_result[0] is None:
            return None

        transitions, indices, weights = sample_result
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        if non_final_mask.any():
            non_final_next_states = torch.stack(
                [s for s in batch.next_state if s is not None]
            ).to(self.device)
        else:
            non_final_next_states = torch.empty(0, 8, 8, 12).to(self.device)

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(
            batch.reward, device=self.device, dtype=torch.float32
        )
        player_batch = torch.tensor(
            batch.player, device=self.device, dtype=torch.float32
        )
        weights = weights.to(self.device)

        # Current Q values
        current_q_values = self.model(state_batch, turn=player_batch)
        state_action_values = current_q_values.gather(1, action_batch)

        # Next Q values using Double DQN
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        if non_final_mask.any():
            with torch.no_grad():
                # Use main network to select actions
                next_actions = (
                    self.model(non_final_next_states, player_batch[non_final_mask])
                    .max(1)[1]
                    .unsqueeze(1)
                )
                # Use target network to evaluate actions
                next_state_values[non_final_mask] = (
                    self.target_network(
                        non_final_next_states, player_batch[non_final_mask]
                    )
                    .gather(1, next_actions)
                    .squeeze()
                )

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute TD errors for priority updates
        td_errors = torch.abs(
            state_action_values.squeeze() - expected_state_action_values
        ).detach()

        # Compute weighted loss
        loss = self.criterion(
            state_action_values.squeeze(), expected_state_action_values
        )
        weighted_loss = (loss * weights).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities in replay buffer
        new_priorities = (
            td_errors.cpu().numpy() + 1e-6
        )  # Small epsilon to avoid zero priorities
        self.memory.update_priorities(indices, new_priorities)

        return weighted_loss.item()

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

        # Handle backward compatibility: convert old ReplayMemory to PrioritizedReplayMemory
        if hasattr(self, "memory") and isinstance(self.memory, ReplayMemory):
            rprint(
                "[bold yellow]Converting old ReplayMemory to PrioritizedReplayMemory...[/bold yellow]"
            )
            old_memory = self.memory
            self.memory = PrioritizedReplayMemory(self.REPLAY_MEMORY_SIZE)
            # Copy existing transitions if any
            for transition in old_memory.memory:
                if transition is not None:
                    # Add default priority to old transitions
                    self.memory.push(
                        *transition[:6]
                    )  # Exclude old priority if it exists

    def fit(self, env: gym.Env) -> None:
        # Print configuration at start
        rprint("[bold blue]Starting Chess RL Training[/bold blue]")
        self.print_config("Training Configuration")

        # Initialize wandb in offline mode to avoid login prompts
        wandb.init(project="chess-rl", config=self.config, mode="offline")
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon = epsilon_start

        if not self.continue_from_checkpoint_path:
            self.create_q_model(action_space=env.action_space.n)
        else:
            self.load_agent(self.continue_from_checkpoint_path)
            rprint(
                f"[bold blue]Loaded checkpoint from: {self.continue_from_checkpoint_path}[/bold blue]"
            )

        # Initialize metrics tracking
        recent_rewards = deque(maxlen=100)
        recent_losses = deque(maxlen=100)
        recent_game_lengths = deque(maxlen=100)
        wins = 0
        total_games = 0

        # Create rich progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task(
                "[bold green]Training Chess Agent...", total=self.NUMBER_EPISODES
            )

            # Create metrics table
            metrics_table = Table(title="Training Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="magenta")

            for episode_idx in range(self.NUMBER_EPISODES):
                # Handle new environment interface
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    state, info = reset_result
                else:
                    state = reset_result

                done = False
                total_reward = 0
                moves = 0
                episode_losses = []

                while not done:
                    player = moves % 2  # 0 for white, 1 for black

                    if random.random() < epsilon:
                        move = self.explore(env)
                        move_idx = env.move_str_to_index(str(move))
                    else:
                        move, move_idx = self.predict(state, env, turn=player)

                    # Handle new environment interface
                    step_result = env.step(move)
                    if len(step_result) == 5:
                        next_state, reward, done, truncated, info = step_result
                    else:
                        next_state, reward, done = step_result
                        truncated = False

                    # Enhanced reward shaping for chess
                    shaped_reward = self._shape_reward(reward, info, done, moves)

                    self.memory.push(
                        torch.tensor(state, dtype=torch.float32),
                        move_idx,
                        shaped_reward,
                        torch.tensor(next_state, dtype=torch.float32)
                        if not done
                        else None,
                        done,
                        player,
                    )

                    state = next_state
                    total_reward += shaped_reward
                    moves += 1

                    if moves % self.OPTIMIZE_EVERY_N_STEPS == 0:
                        loss = self.optimize_model()
                        if loss is not None:
                            episode_losses.append(loss)
                            wandb.log({"step_loss": loss, "step": moves})

                    if moves % self.TARGET_UPDATE == 0:
                        self.update_target_network()

                # Update epsilon
                epsilon = max(epsilon_end, epsilon * self.EPSILON_DECAY)
                self.scheduler.step()

                # Track metrics
                recent_rewards.append(total_reward)
                recent_game_lengths.append(moves)
                if episode_losses:
                    avg_episode_loss = np.mean(episode_losses)
                    recent_losses.append(avg_episode_loss)

                # Determine if this was a win (simplified)
                if done and total_reward > 0:
                    wins += 1
                total_games += 1

                # Calculate running averages
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                avg_loss = np.mean(recent_losses) if recent_losses else 0
                avg_game_length = (
                    np.mean(recent_game_lengths) if recent_game_lengths else 0
                )
                win_rate = (wins / total_games) * 100 if total_games > 0 else 0

                # Log to wandb
                wandb_metrics = {
                    "episode": episode_idx,
                    "total_reward": total_reward,
                    "avg_reward_100": avg_reward,
                    "moves": moves,
                    "avg_game_length_100": avg_game_length,
                    "epsilon": epsilon,
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "win_rate_100": win_rate,
                    "memory_size": len(self.memory),
                }

                if episode_losses:
                    wandb_metrics["avg_episode_loss"] = avg_episode_loss
                    wandb_metrics["avg_loss_100"] = avg_loss

                wandb.log(wandb_metrics)

                # Update progress bar with detailed info
                progress.update(
                    main_task,
                    advance=1,
                    description=f"[bold green]Episode {episode_idx + 1}/{self.NUMBER_EPISODES} | "
                    f"Reward: {total_reward:.2f} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Win Rate: {win_rate:.1f}% | "
                    f"ε: {epsilon:.4f}",
                )

                # Print detailed metrics every 10 episodes
                if (episode_idx + 1) % 10 == 0:
                    metrics_table = Table(
                        title=f"Training Metrics - Episode {episode_idx + 1}"
                    )
                    metrics_table.add_column("Metric", style="cyan")
                    metrics_table.add_column("Value", style="magenta")

                    metrics_table.add_row(
                        "Episode", f"{episode_idx + 1}/{self.NUMBER_EPISODES}"
                    )
                    metrics_table.add_row("Total Reward", f"{total_reward:.2f}")
                    metrics_table.add_row("Avg Reward (100)", f"{avg_reward:.2f}")
                    metrics_table.add_row("Game Length", f"{moves}")
                    metrics_table.add_row(
                        "Avg Game Length (100)", f"{avg_game_length:.1f}"
                    )
                    metrics_table.add_row("Win Rate (100)", f"{win_rate:.1f}%")
                    metrics_table.add_row("Epsilon", f"{epsilon:.4f}")
                    metrics_table.add_row(
                        "Learning Rate", f"{self.scheduler.get_last_lr()[0]:.6f}"
                    )
                    metrics_table.add_row("Memory Size", f"{len(self.memory)}")

                    if episode_losses:
                        metrics_table.add_row(
                            "Avg Episode Loss", f"{avg_episode_loss:.6f}"
                        )
                        metrics_table.add_row("Avg Loss (100)", f"{avg_loss:.6f}")

                    console.print(metrics_table)

                # Save checkpoints
                if episode_idx % self.save_every_n_episodes == 0:
                    self.save_agent(f"checkpoints/episode_{episode_idx}.ckpt")
                    self.save_agent(f"checkpoints/last.ckpt")
                    rprint(
                        f"[bold yellow]Checkpoint saved at episode {episode_idx}[/bold yellow]"
                    )

        rprint("[bold green]Training completed![/bold green]")
        wandb.finish()

    def _shape_reward(self, base_reward, info, done, moves):
        """Enhanced reward shaping for chess training"""
        shaped_reward = base_reward

        # Encourage longer games (avoid quick losses)
        if done and moves < 10:
            shaped_reward -= 5.0

        # Bonus for check
        if info.get("is_check", False):
            shaped_reward += 0.5

        # Small penalty for very long games to encourage decisive play
        if moves > 100:
            shaped_reward -= 0.01

        return shaped_reward

    def print_config(self, title="Agent Configuration"):
        """Print the agent configuration in a nice table format"""
        config_table = Table(title=title)
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="magenta")

        config_table.add_row("Gamma (Discount Factor)", f"{self.gamma}")
        config_table.add_row("Learning Rate", f"{self.LEARNING_RATE}")
        config_table.add_row("Batch Size", f"{self.BATCH_SIZE}")
        config_table.add_row("Number of Episodes", f"{self.NUMBER_EPISODES}")
        config_table.add_row("Replay Memory Size", f"{self.REPLAY_MEMORY_SIZE}")
        config_table.add_row("Optimize Every N Steps", f"{self.OPTIMIZE_EVERY_N_STEPS}")
        config_table.add_row("Target Update Frequency", f"{self.TARGET_UPDATE}")
        config_table.add_row("Epsilon Decay", f"{self.EPSILON_DECAY}")
        config_table.add_row("Save Every N Episodes", f"{self.save_every_n_episodes}")
        config_table.add_row("Device", f"{self.device}")

        if self.continue_from_checkpoint_path:
            config_table.add_row(
                "Checkpoint Path", f"{self.continue_from_checkpoint_path}"
            )

        console.print(config_table)
