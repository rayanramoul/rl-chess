import random
from collections import namedtuple
import pickle
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DeepQAgent(nn.Module):
    def __init__(self, model=None, gamma=0.9, lr=0.001, list_of_moves=None):
        super(DeepQAgent, self).__init__()
        self.gamma = gamma
        self.number_of_actions = len(list_of_moves)
        self.list_of_moves = list_of_moves
        if model:
            # Self Model is the Policy Network
            self.model = model
            self.target_network = model.copy()
        else:
            self.model = self.create_q_model()
            self.model = self.create_q_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.actions_history = []
        self.rewards_history = []
        self.states_history = []
        
        self.memory = ReplayMemory(10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.BATCH_SIZE = 128
        self.criterion = nn.MSELoss() # F.mse_loss
        
    def create_q_model(self):
        """
        
        Input shape is of shape [1, 8, 8, 1]
        or [batch_dim, height(of the chess board), width(of the chess booard), channels]
        """
        # Network defined by the Deepmind paper
        """
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
        )"""
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=2 * 2, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.number_of_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device).unsqueeze(0)
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
        state = env.translate_board()
        print("Exploitation state shape : ", state.shape)
        action_probs = self.forward(state)
        print("Self model: \n", self.model)
        print("Action probs: \n", action_probs.shape)
        move_number = torch.argmax(torch.tensor(action_probs), dim=0)
        print("Move number: ", move_number)
        print("Len list of moves : ", len(self.list_of_moves))
        move_str = self.list_of_moves[move_number]
        return move_str, move_number

    def optimize_model(self):
        """
        Fonction effectuant les étapes de mise à jour du réseau ( sampling des épisodes, calcul de loss, rétro propagation )
        """
        # Si la taille actuelle de notre mémoire est inférieure aux batchs de mémoires qu'on veut prendre en compte on n'effectue
        # Pas encore d'optimization
        if len(self.memory) < self.BATCH_SIZE:
            return

        # Extraction d'un echantillon aléatoire de la mémoire ( ou chaque éléments est constitué de (état, nouvel état, action, récompense) )
        # Et ce pour éviter le biais occurant si on apprenait sur des états successifs
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # Séparation des différents éléments contenus dans les différents echantillons
        non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state))).bool()
        next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = Variable(torch.cat(next_states), 
                                         volatile=True).type(torch.Tensor)
        
        batch_state = [s for s in batch.state if s is not None]
        # print("batch state : ", batch_state)
        batch_state = [torch.tensor(s, dtype=torch.float).unsqueeze(0) for s in batch_state]
        # print("State batch before cat : ", batch_state.shape)
        print("len set state batch : ", len(batch.state))
        
        
        state_batch = Variable(torch.cat(batch_state, dim=0)).type(torch.Tensor)
        # swap dim 0 and 3 of state_batch
        # state_batch = torch.view_as_real(state_batch)
        print("State batch after cat : ", state_batch.shape)
        state_batch = state_batch.reshape(128, 1, 8, 8)
        print("State batch after RESHAPE : ", state_batch.shape)
        if self.device == 'cuda': # use_cuda:
            state_batch = state_batch.cuda()
        action_batch = Variable(torch.LongTensor(batch.action).view(-1,1)).type(torch.LongTensor)
        reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1,1)).type(torch.Tensor)


        # Passage des états par le Q-Network ( en calculate Q(s_t, a) ) et on récupére les actions sélectionnées
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Calcul de V(s_{t+1}) pour les prochain états.
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE, 1).type(torch.Tensor)) 

        if self.device == 'cuda':
            non_final_next_states = non_final_next_states.cuda()
        
        # Appel au second Q-Network ( celui de copie pour garantir la stabilité de l'apprentissage )
        d = self.target_net(non_final_next_states) 
        next_state_values[non_final_mask] = d.max(1)[0].view(-1,1)
        next_state_values.volatile = False

        # On calcule les valeurs de fonctions Q attendues ( en faisant appel aux récompenses attribuées )
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Calcul de la loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        # Rétro-propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_net = copy.deepcopy(self.model)  
    
    def train(self, state, move_number, reward, next_state, done):
        """
        Train the agent network
        """
        target_q = reward
        if not done:
            processed_state = self.forward(next_state)
            target_q += torch.Tensor(self.gamma * torch.max(processed_state))

        current_q = self.forward(state)
        current_q = current_q[:, move_number]
        target_q = torch.Tensor([target_q])
        # loss = F.mse_loss(current_q, target_q)
        
        self.memory.push(torch.Tensor(state), int(move_number), torch.Tensor(next_state), reward)
        self.optimize_model()
        
    def save(self, folder_path:str):
        torch.save(self.model.state_dict(), os.path.join(folder_path, 'model.pth'))

        # pickle save self
        pickle.dump(self, open(os.path.join(folder_path, 'deep_q_agent.pickle'), 'wb'))
        
