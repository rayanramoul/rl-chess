import random
import pickle


class Agent:
    def __init__(self):
        # set seed random
        # random.seed(0)
        pass

    def save_pickle_agent(self, path):
        pickle.dump(self, open(path, "wb"))

    def initialize(self):
        print("Agent initialized")

    def choose_movement(self, state_board, possible_movements):
        chosen_movement = random.choice(list(possible_movements))
        return chosen_movement
