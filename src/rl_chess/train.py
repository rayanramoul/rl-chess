import hydra
from omegaconf import DictConfig
import os
import torch

from rl_chess.agents.deep_q_agent import DeepQAgent
from rl_chess.environements.chess_env import ChessEnv
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    from pprint import pprint

    print("Training Configuration:")
    pprint(cfg)

    # Choose training method
    training_method = cfg.get("training_method", "dqn")  # default to DQN

    if training_method == "dqn":
        train_with_dqn(cfg)
    elif training_method == "ppo":
        train_with_ppo(cfg)
    else:
        raise ValueError(f"Unknown training method: {training_method}")


def train_with_dqn(cfg: DictConfig):
    """Train using custom Deep Q-Network implementation"""
    print("Training with Deep Q-Network (DQN)...")

    # Create environment and agent
    env = ChessEnv(**cfg.env)
    agent = DeepQAgent(number_of_actions=env.number_of_possible_moves, **cfg.agent)

    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)

    # Train the agent
    agent.fit(env=env)

    print("DQN Training completed!")


def train_with_ppo(cfg: DictConfig):
    """Train using Ray RLlib PPO"""
    print("Training with Proximal Policy Optimization (PPO)...")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Configure the PPO algorithm
    config = (
        PPOConfig()
        .environment(ChessEnv)
        .framework("torch")
        .training(
            gamma=cfg.agent.get("gamma", 0.99),
            lr=cfg.agent.get("lr", 0.0003),
            kl_coeff=0.3,
            train_batch_size=cfg.agent.get("batch_size", 4000),
            sgd_minibatch_size=cfg.agent.get("sgd_minibatch_size", 128),
            num_sgd_iter=cfg.agent.get("num_sgd_iter", 10),
        )
        .rollouts(
            num_rollout_workers=cfg.get("num_workers", 2),
            rollout_fragment_length=cfg.get("rollout_fragment_length", 200),
        )
        .evaluation(
            evaluation_interval=cfg.get("evaluation_interval", 10),
            evaluation_num_workers=1,
            evaluation_duration=5,
        )
        .resources(
            num_gpus=1 if torch.cuda.is_available() else 0,
        )
    )

    # Train the agent
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=ray.air.RunConfig(
            stop={
                "training_iteration": cfg.agent.get("number_episodes", 100),
                "timesteps_total": cfg.agent.get("max_timesteps", 1000000),
            },
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_frequency=cfg.agent.get("save_every_n_episodes", 10)
            ),
            name="chess_ppo_training",
        ),
    )

    results = tuner.fit()

    # Get the best result
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"Best training result: {best_result}")

    print("PPO Training completed!")


def train_with_dqn_rllib(cfg: DictConfig):
    """Train using Ray RLlib DQN (alternative implementation)"""
    print("Training with Ray RLlib DQN...")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Configure the DQN algorithm
    config = (
        DQNConfig()
        .environment(ChessEnv)
        .framework("torch")
        .training(
            gamma=cfg.agent.get("gamma", 0.99),
            lr=cfg.agent.get("lr", 0.0005),
            train_batch_size=cfg.agent.get("batch_size", 32),
            target_network_update_freq=cfg.agent.get("target_update", 1000),
            replay_buffer_config={
                "capacity": cfg.agent.get("replay_memory_size", 100000),
            },
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.1,
                "epsilon_timesteps": cfg.agent.get("epsilon_timesteps", 100000),
            }
        )
        .rollouts(
            num_rollout_workers=cfg.get("num_workers", 2),
        )
        .evaluation(
            evaluation_interval=cfg.get("evaluation_interval", 10),
            evaluation_num_workers=1,
            evaluation_duration=5,
        )
        .resources(
            num_gpus=1 if torch.cuda.is_available() else 0,
        )
    )

    # Train the agent
    tuner = tune.Tuner(
        "DQN",
        param_space=config.to_dict(),
        run_config=ray.air.RunConfig(
            stop={
                "training_iteration": cfg.agent.get("number_episodes", 100),
                "timesteps_total": cfg.agent.get("max_timesteps", 1000000),
            },
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_frequency=cfg.agent.get("save_every_n_episodes", 10)
            ),
            name="chess_dqn_training",
        ),
    )

    results = tuner.fit()

    # Get the best result
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(f"Best training result: {best_result}")

    print("DQN Training completed!")


if __name__ == "__main__":
    main()
