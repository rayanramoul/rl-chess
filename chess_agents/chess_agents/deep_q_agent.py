import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

chess_dict = {
    'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
    'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
    'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
    'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
    'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
    'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
    'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
    'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
    'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
    'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
    'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
    'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
    '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
}

        


def translate_board(board): 
    pgn = board.epd()
    foo = []  
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        foo2 = []  
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo2.append(chess_dict['.'])
            else:
                foo2.append(chess_dict[thing])
        foo.append(foo2)
    return np.array(foo)

def board_matrix(board): 
    pgn = board.epd()
    foo = []  
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")
    for row in rows:
        foo2 = []  
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo.append('.')
            else:
                foo.append(thing)
    return np.array(foo)

def translate_move(move):
    from_square = move.from_square
    to_square = move.to_square
    return np.array([from_square,to_square])




class DeepQAgent(nn.Module):
    def __init__(self, model=None, gamma=0.9, lr=0.001, list_of_moves=None):
        super(DeepQAgent, self).__init__()
        self.gamma = gamma
        self.number_of_actions = len(list_of_moves)
        self.list_of_moves = list_of_moves
        if model:
            self.model = model
        else:
            self.model = self.create_q_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.actions_history = []
        self.rewards_history = []
        self.states_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def create_q_model(self):
        # Network defined by the Deepmind paper
        return nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, self.number_of_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        return self.model(x)

    def predict(self, env):
        state_tensor = torch.tensor(env.translate_board()).unsqueeze(0)
        action_probs = self.model(state_tensor)
        move_number = torch.argmax(torch.tensor(action_probs), dim=None)
        move = self.list_of_moves[move_number]
        return move, move_number

    def explore(self, env):
        # Modify this function to return valid action for your env
        # get legal moves from board env and return random move
        move_number = random.randint(0, len(self.list_of_moves) - 1)
        move_str = self.list_of_moves[move_number]
        return move_str, move_number

    def exploit(self, env):
        # Modify this function to return valid action for your env
        action_probs = self.forward(env.translate_board())
        move_number = torch.argmax(torch.tensor(action_probs), dim=None)
        move_str = self.list_of_moves[move_number.item()]
        return move_str, move_number

    def train(self, state, action, reward, next_state, done):
        target_q = reward
        # print("\nstate shape: ", state.shape)
        # print("next state shape: ", next_state.shape)
        if not done:
            processed_state = self.forward(next_state)
            # print("train processed_state: ", processed_state.shape)
            target_q += torch.Tensor(self.gamma * torch.max(processed_state))

        # print("train action: ", action)
        current_q = self.forward(state)
        # print("train current q shape : ", current_q.shape)
        current_q = current_q[:, action]
        # print("train current q shape : ", current_q.shape)
        target_q = torch.Tensor([target_q])
        # print("train target q shape : ", target_q)
        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
