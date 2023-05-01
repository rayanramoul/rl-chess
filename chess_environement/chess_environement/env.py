import chess
from gym import spaces, error, utils, Env
from gym.utils import seeding


switch={
            'p':10,
            'P':-10,
            'q':90,
            'Q':-90,
            'n':30,
            'N':-30,
            'r':50,
            'R':-50,
            'b':30,
            'B':-30,
            'k':900,
            'K':-900,
            'None':0
}


class Piece:
    def __init__(self, color, piece_type, position):
        self.color = color
        self.piece_type = piece_type
        self.position = position

    def __str__(self):
        return self.color + " " + self.piece_type + " " + self.position
    

class ChessEnv(Env):
    def __init__(self):
        self.board = chess.Board()
        self.pieces = []
        self.turn = 0
        self.winner = None
        self.legal_moves = []
        self.legal_moves_dict = {}
        self.state = []
        self.reward = 0
        self.done = False
        self.info = {}
    
    def reset(self):
        self.board = chess.Board() 
    
    def step(self, action):
        pass  
    
    def info(self):
        pass   
    
    @property
    def state(self): 
        pass 
    
    @state.setter
    def state(self, state):
        pass
    
    
    @property
    def possible_actions(self):
        self.board.legal_moves
    
    
"