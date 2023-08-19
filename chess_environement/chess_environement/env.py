import gym
from gym import spaces
import chess
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

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        print("Really ?")
        # Define the observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(64,))  # 8x8 board representation

        # Define the action space
        self.action_space = spaces.Discrete(4096)  # 64*64 possible moves

        # Create a Chess board object
        self.board = chess.Board()
        
        self.list_of_moves = self._list_of_moves()
        print("list of moves", len(self.list_of_moves))

    def reset(self):
        # Reset the board to the initial state
        self.board.reset()

        # Convert the board position to an observation
        observation = self._get_observation()

        return observation

    def step(self, action):
        # Convert the action to a move on the board
        move = self._action_to_move(action)

        # Make the move on the board
        self.board.push(move)

        # Convert the board position to an observation
        observation = self._get_observation()

        # Get the reward and check if the game is over
        reward = self._get_reward()
        done = self.board.is_game_over()

        return observation, reward, done, {}

    def render(self, mode='human'):
        print(self.board)

    def _get_observation(self):
        # Convert the board position to a binary observation
        observation = []

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                observation.append(1)  # Piece present
                observation.extend(self._encode_piece(piece))
            else:
                observation.append(0)  # Empty square

        return observation

    def _get_reward(self):
        # Simple reward function example
        if self.board.is_checkmate():
            return 1  # Win
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.5  # Draw
        else:
            return 0  # No reward

    def _action_to_move(self, action):
        # Convert action index to a move
        moves = list(self.board.legal_moves)
        move = moves[action]
        return move

    def _list_of_moves(self):
        self.board_all_moves = []
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        for x in letters:
            for y in range(1, 9):
                for x1 in letters:
                    for y1 in range(1, 9):
                        self.board_all_moves.append(f"{x}{y}{x1}{y1}")
        # add the promotion moves
        for x in letters:
            for y in [8, 1]:
                for p in ['q', 'r', 'b', 'n']:
                    diff_y = 1 if y == 1 else -1
                    self.board_all_moves.append(f"{x}{y}{x}{diff_y}{p}")
                        
            
    def _encode_piece(self, piece):
        # Encode the piece type using a one-hot encoding
        piece_types = ['p', 'r', 'n', 'b', 'q', 'k']
        encoding = [0] * 6
        encoding[piece_types.index(piece.symbol().lower())] = 1
        return encoding
    
    
    def translate_board(self):
        return translate_board(self.board)


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
  