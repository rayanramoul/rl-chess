import gym
from gym import spaces
import chess

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()

        # Define the observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(64,))  # 8x8 board representation

        # Define the action space
        self.action_space = spaces.Discrete(4096)  # 64*64 possible moves

        # Create a Chess board object
        self.board = chess.Board()

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

    def _encode_piece(self, piece):
        # Encode the piece type using a one-hot encoding
        piece_types = ['p', 'r', 'n', 'b', 'q', 'k']
        encoding = [0] * 6
        encoding[piece_types.index(piece.symbol().lower())] = 1
        return encoding