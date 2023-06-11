import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import chess
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

value_dict = {
    'p': [1, 0],
    'P': [0, 1],
    'n': [3, 0],
    'N': [0, 3],
    'b': [3, 0],
    'B': [0, 3],
    'r': [5, 0],
    'R': [0, 5],
    'q': [9, 0],
    'Q': [0, 9],
    'k': [0, 0],
    'K': [0, 0],
    '.': [0, 0]
}

num2move = {}
move2num = {}
counter = 0
for from_sq in range(64):
    for to_sq in range(64):
        num2move[counter] = chess.Move(from_sq,to_sq)
        move2num[chess.Move(from_sq,to_sq)] = counter
        counter += 1
        
def generate_side_matrix(board,side):
    matrix = board_matrix(board)
    translate = translate_board(board)
    bools = np.array([piece.isupper() == side for piece in matrix])
    bools = bools.reshape(8,8,1)
    
    side_matrix = translate*bools
    return np.array(side_matrix)

def generate_input(positions,len_positions = 8):
    board_rep = []
    for position in positions:
        black = generate_side_matrix(position,False)
        white = generate_side_matrix(position,True)
        board_rep.append(black)
        board_rep.append(white)
    turn = np.zeros((8,8,12))
    turn.fill(int(position.turn))
    board_rep.append(turn)
    
    while len(board_rep) < len_positions*2 + 1:
        value = np.zeros((8,8,12))
        board_rep.insert(0,value)
    board_rep = np.array(board_rep)
    return board_rep

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

def filter_legal_moves(board,logits):
    filter_mask = np.zeros(logits.shape)
    legal_moves = board.legal_moves
    for legal_move in legal_moves:
        from_square = legal_move.from_square
        to_square = legal_move.to_square
        idx = move2num[chess.Move(from_square,to_square)]
        filter_mask[idx] = 1
    new_logits = logits*filter_mask
    return new_logits


class DeepQAgent(nn.Module):
    def __init__(self, model=None, gamma=0.9, lr=0.001):
        super(DeepQAgent, self).__init__()
        self.gamma = gamma

        if model:
            print('CUSTOM MODEL SET')
            self.model = model
        else:
            self.model = self.create_q_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
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
        # Modify this function to return valid action for your env
        action_space = np.random.randn(4096)
        action_space = filter_legal_moves(env.board, action_space)
        action = np.argmax(action_space, axis=None)
        move = num2move[action]
        return move, action

    def exploit(self, env):
        # Modify this function to return valid action for your env
        action_probs = self.forward(env.translate_board())
        action_space = filter_legal_moves(env.board, action_probs[0].detach().numpy())
        action = torch.argmax(torch.tensor(action_space), dim=None)
        move = num2move[action.item()]
        return move, action

    def train(self, state, action, reward, next_state, done):
        target_q = reward
        if not done:
            target_q += self.gamma * torch.max(self.forward(next_state))

        current_q = self.forward(state)[action]

        loss = F.mse_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
