import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DeepQAgent(nn.Module):
    def __init__(self, model=None):
        super(DeepQAgent, self).__init__()
        if model:
            print('CUSTOM MODEL SET')
            self.model = model
        else:
            self.model = self.create_q_model()
        self.actions_history = []
        self.rewards_history = []
        self.states_history = []

    def create_q_model(self):
        # Network defined by the Deepmind paper
        return nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 4096),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, env):
        state_tensor = torch.tensor(env.translate_board()).unsqueeze(0)
        action_probs = self.model(state_tensor)
        action_space = filter_legal_moves(env.board, action_probs[0].detach().numpy())
        action = torch.argmax(torch.tensor(action_space), dim=None)
        move = num2move[action.item()]
        return move, action

    def explore(self, env):
        action_space = np.random.randn(4096)
        action_space = filter_legal_moves(env.board, action_space)
        action = np.argmax(action_space, axis=None)
        move = num2move[action]
        return move, action
