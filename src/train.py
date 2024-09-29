import hydra
from omegaconf import DictConfig

from src.agents.deep_q_agent import DeepQAgent
from src.environements.chess_env import ChessEnv

# Initialize environment and agent


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    from pprint import pprint

    pprint(cfg)

    env = ChessEnv(**cfg.env)
    agent = DeepQAgent(number_of_actions=env.number_of_possible_moves, **cfg.agent)
    metrics: dict = {}
    agent.fit(env=env)


if __name__ == "__main__":
    main()
