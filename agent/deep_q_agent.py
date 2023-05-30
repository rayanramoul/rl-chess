import random
import torch
from torch import nn
from torch.nn import functional as F

class module(nn.Module):
    def __init__(self, hidden_size):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SeLU()
        self.activation2 = nn.SeLU()
    
    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_input
        x = self.activation2(x)
        return x

class DeeQAgent(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200):
        # set seed random
        # random.seed(0)
        self.hidden_layers = hidden_layers
        self.input_layer =  nn.Conv2d(6, hidden_size, kernel_size=3, stride=1, padding=1)
        self.module_1 = nn.ModuleList([module(hidden_size) for i  in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)
        print("\nDEEP-Q AGENT CREATED\n")
        
        
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        
        for i in range(self.hidden_layers):
            x = self.module_1[i](x)
        x = self.output_layer(x)
        return x
        
    def choose_movement(self, state_board, possible_movements):
        # choose random movement
        print("possible movements : ", possible_movements)
        chosen_movement = random.choice(list(possible_movements))
        print("chosen movement : ", chosen_movement)
        return chosen_movement