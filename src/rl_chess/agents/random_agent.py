import random
from chess_agents.chess_agents.agent import Agent


class RandomAgent(Agent):
    def __init__(self):
        # set seed random
        # random.seed(0)
        pass

    def initialize(self):
        print("Agent initialized")

    def choose_movement(self, state_board, possible_movements):
        # choose random movement
        chosen_movement = random.choice(list(possible_movements))
        return chosen_movement
