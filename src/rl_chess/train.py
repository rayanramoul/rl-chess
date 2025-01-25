import hydra
from omegaconf import DictConfig

from rl_chess.agents.deep_q_agent import DeepQAgent
from rl_chess.environements.chess_env import ChessEnv
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

# Initialize environment and agent


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    from pprint import pprint

    pprint(cfg)

    # env = ChessEnv(**cfg.env)
    # agent = DeepQAgent(number_of_actions=env.number_of_possible_moves, **cfg.agent)
    # metrics: dict = {}
    # agent.fit(env=env)
    # Initialize Ray
    ray.init()

    # Configure the algorithm
    config = (
        PPOConfig()
        .environment(ChessEnv)
        .framework("torch")
        .training(
            gamma=0.99,
            lr=0.0003,
            kl_coeff=0.3,
            train_batch_size=4000,
        )
        .evaluation(evaluation_interval=10)
    )

    # Train the agent
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=ray.air.RunConfig(
            stop={"training_iteration": 100},
            checkpoint_config=ray.air.CheckpointConfig(checkpoint_frequency=10),
        ),
    )

    results = tuner.fit()


if __name__ == "__main__":
    main()
