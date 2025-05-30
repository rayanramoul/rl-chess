from loguru import logger
import gymnasium as gym
from gymnasium import spaces
import chess
from typing import Optional, Tuple, Dict, Any

import numpy as np

chess_dict = {
    "p": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "P": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "n": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "N": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "b": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "B": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "r": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "R": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "q": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "Q": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "k": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "K": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ".": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

value_dict = {
    "p": [1, 0],
    "P": [0, 1],
    "n": [3, 0],
    "N": [0, 3],
    "b": [3, 0],
    "B": [0, 3],
    "r": [5, 0],
    "R": [0, 5],
    "q": [9, 0],
    "Q": [0, 9],
    "k": [0, 0],
    "K": [0, 0],
    ".": [0, 0],
}


class ChessEnv(gym.Env):
    """Chess Environment based on Gym specifications."""

    def __init__(self, env_name="ChessEnv", render_mode: str | None = None):
        super(ChessEnv, self).__init__()
        self.env_name = env_name

        # Create a Chess board object
        self.board = chess.Board()

        # Create all possible moves mapping
        self.all_possibles_moves_list = self._create_moves_mapping()
        self.inversed_all_possibles_moves_list = {
            v: k for k, v in self.all_possibles_moves_list.items()
        }

        # Define action space as discrete with all possible chess moves
        self._action_space = spaces.Discrete(len(self.all_possibles_moves_list))

        # Observation space: 8x8 board with 12 channels (6 piece types x 2 colors)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, 8, 12), dtype=np.float32
        )

    @property
    def action_space(self):
        """Return the action space for RLlib compatibility"""
        return self._action_space

    @property
    def number_of_possible_moves(self):
        return len(self.all_possibles_moves_list)

    def _create_moves_mapping(self):
        """Create mapping of all possible chess moves"""
        moves = []
        letters = ["a", "b", "c", "d", "e", "f", "g", "h"]

        # Regular moves
        for x in letters:
            for y in range(1, 9):
                for x1 in letters:
                    for y1 in range(1, 9):
                        if x == x1 and y == y1:
                            continue
                        moves.append(f"{x}{y}{x1}{y1}")

        # Promotion moves
        for x in letters:
            for x2 in letters:
                for y in [7, 2]:  # Second-to-last rank for both colors
                    for p in ["q", "r", "b", "n"]:
                        diff_y = 1 if y == 2 else -1
                        moves.append(f"{x}{y}{x2}{y + diff_y}{p}")

        return {move: idx for idx, move in enumerate(moves)}

    def reset(self, seed=None, options=None):
        """Reset the board to the initial state"""
        if seed is not None:
            np.random.seed(seed)

        # Reset the board to the initial state
        self.board.reset()
        observation = self._get_observation()

        info = {
            "legal_moves": list(self.board.legal_moves),
            "is_check": self.board.is_check(),
            "turn": "white" if self.board.turn else "black",
        }

        return observation, info

    def _get_observation(self):
        """Convert current board state to observation"""
        observation = np.zeros((8, 8, 12), dtype=np.float32)

        piece_mapping = {
            "p": 0,
            "n": 1,
            "b": 2,
            "r": 3,
            "q": 4,
            "k": 5,
            "P": 6,
            "N": 7,
            "B": 8,
            "R": 9,
            "Q": 10,
            "K": 11,
        }

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                rank, file = chess.square_rank(square), chess.square_file(square)
                piece_idx = piece_mapping[piece.symbol()]
                observation[rank, file, piece_idx] = 1

        return observation

    @property
    def state(self):
        return self._get_observation()

    def move_str_to_index(self, move):
        move_str = str(move)
        if move_str in self.all_possibles_moves_list:
            return self.all_possibles_moves_list[move_str]
        else:
            # Handle case where move is not in our predefined list
            # This shouldn't happen with legal moves, but just in case
            return 0

    def move_index_to_str(self, move):
        if isinstance(move, tuple):
            move = move[0]
        return self.inversed_all_possibles_moves_list.get(move, "a1a1")

    def step(self, action):
        """Execute action and return new state, reward, done, truncated, and info"""

        # Convert action index to move string
        if isinstance(action, int):
            move_str = self.move_index_to_str(action)
        else:
            move_str = str(action)

        try:
            move = chess.Move.from_uci(move_str)

            # Check if move is legal
            if move not in self.board.legal_moves:
                # Return negative reward for illegal move and end episode
                return (
                    self._get_observation(),
                    -10.0,
                    True,
                    False,
                    {"illegal_move": True, "move": move_str},
                )

            # Make the move
            self.board.push(move)

        except (ValueError, chess.InvalidMoveError):
            # Invalid move format
            return (
                self._get_observation(),
                -10.0,
                True,
                False,
                {"invalid_move": True, "move": move_str},
            )

        # Calculate reward
        reward = self._calculate_reward()

        # Check if game is over
        done = self.board.is_game_over()

        # Get observation and info
        observation = self._get_observation()
        info = {
            "legal_moves": list(self.board.legal_moves),
            "is_check": self.board.is_check(),
            "turn": "white" if self.board.turn else "black",
            "move": move_str,
        }

        return observation, reward, done, False, info

    def _calculate_reward(self) -> float:
        """Calculate reward based on game state"""
        if self.board.is_checkmate():
            # Reward based on whose turn it is (the losing player)
            return 100.0 if not self.board.turn else -100.0

        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0

        # Small reward for check
        if self.board.is_check():
            return 1.0 if self.board.turn else -1.0

        # Material advantage calculation
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None and piece.piece_type != chess.KING:
                if piece.color:  # White
                    white_material += piece_values[piece.piece_type]
                else:  # Black
                    black_material += piece_values[piece.piece_type]

        # Return material advantage as a small reward
        material_diff = (white_material - black_material) * 0.01
        return material_diff if self.board.turn else -material_diff

    def render(self, mode="human"):
        """Render the current board state"""
        if mode == "human":
            print(self.board)

    @property
    def available_moves(self):
        available_moves = list(self.board.legal_moves)
        return available_moves

    def _action_to_move(self, action):
        # Convert action index to a move
        moves = list(self.board.legal_moves)
        if action < len(moves):
            return moves[action]
        else:
            return moves[0] if moves else None

    def _encode_piece(self, piece):
        # Encode the piece type using a one-hot encoding
        piece_types = ["p", "r", "n", "b", "q", "k"]
        encoding = [0] * 6
        encoding[piece_types.index(piece.symbol().lower())] = 1
        return encoding

    def translate_board(self):
        return translate_board(self.board)


def translate_board(board):
    numerical_board = [[0] * 8 for _ in range(8)]  # Initialize an 8x8 grid with zeros

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,  # Assigning 0 for simplicity, but you can assign a value if needed
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_value = piece_values[piece.piece_type]
            if piece.color == chess.BLACK:
                piece_value = -piece_value  # Invert value for black pieces
            rank, file = chess.square_rank(square), chess.square_file(square)
            numerical_board[7 - rank][file] = piece_value

    return np.array(numerical_board)


class ChessGymEnv(gym.Env):
    """Chess environment following gym interface"""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, env_name="ChessGymEnv", render_mode: Optional[str] = None):
        super(ChessGymEnv, self).__init__()

        self.env_name = env_name
        self.render_mode = render_mode

        # Initialize chess board
        self.board = chess.Board()

        # Setup action and observation spaces
        self.all_possible_moves = self._create_moves_mapping()

        # Action space: discrete number of all possible moves
        self.action_space = spaces.Discrete(len(self.all_possible_moves))

        # Observation space: 8x8 board with 12 channels (6 piece types x 2 colors)
        # Each square contains a 12-dimensional one-hot vector
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(8, 8, 12), dtype=np.float32
        )

    def _create_moves_mapping(self):
        """Create mapping of all possible chess moves"""
        moves = []
        letters = ["a", "b", "c", "d", "e", "f", "g", "h"]

        # Regular moves
        for x in letters:
            for y in range(1, 9):
                for x1 in letters:
                    for y1 in range(1, 9):
                        if x == x1 and y == y1:
                            continue
                        moves.append(f"{x}{y}{x1}{y1}")

        # Promotion moves
        for x in letters:
            for x2 in letters:
                for y in [7, 2]:  # Second-to-last rank for both colors
                    for p in ["q", "r", "b", "n"]:
                        diff_y = 1 if y == 2 else -1

                        moves.append(f"{x}{y}{x2}{y + diff_y}{p}")

        return {move: idx for idx, move in enumerate(moves)}

    def _get_observation(self):
        """Convert current board state to observation"""
        observation = np.zeros((8, 8, 12), dtype=np.float32)

        piece_mapping = {
            "p": 0,
            "n": 1,
            "b": 2,
            "r": 3,
            "q": 4,
            "k": 5,
            "P": 6,
            "N": 7,
            "B": 8,
            "R": 9,
            "Q": 10,
            "K": 11,
        }

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                rank, file = chess.square_rank(square), chess.square_file(square)
                piece_idx = piece_mapping[piece.symbol()]
                observation[rank, file, piece_idx] = 1

        return observation

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        self.board.reset()
        observation = self._get_observation()

        info = {
            "legal_moves": list(self.board.legal_moves),
            "is_check": self.board.is_check(),
            "turn": "white" if self.board.turn else "black",
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action and return new state, reward, done, truncated, and info"""

        # Convert action index to move string
        move_str = list(self.all_possible_moves.keys())[action]

        try:
            move = chess.Move.from_uci(move_str)
            # Check if move is legal
            if move not in self.board.legal_moves:
                return (
                    self._get_observation(),
                    -10.0,
                    True,
                    False,
                    {"illegal_move": True},
                )

            # Make the move
            self.board.push(move)

        except ValueError:
            # Invalid move format
            return self._get_observation(), -10.0, True, False, {"invalid_move": True}

        # Calculate reward
        reward = self._calculate_reward()

        # Check if game is over
        done = self.board.is_game_over()

        # Get observation and info
        observation = self._get_observation()
        info = {
            "legal_moves": list(self.board.legal_moves),
            "is_check": self.board.is_check(),
            "turn": "white" if self.board.turn else "black",
        }

        return observation, reward, done, False, info

    def _calculate_reward(self) -> float:
        """Calculate reward based on game state"""
        if self.board.is_checkmate():
            return 100.0 if self.board.turn else -100.0

        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0

        # Material advantage calculation
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        white_material = 0
        black_material = 0

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None and piece.piece_type != chess.KING:
                if piece.color:  # White
                    white_material += piece_values[piece.piece_type]
                else:  # Black
                    black_material += piece_values[piece.piece_type]

        # Return material advantage as a small reward
        return (white_material - black_material) * 0.1

    def render(self):
        """Render the current board state"""
        if self.render_mode == "human":
            print(self.board)

    def close(self):
        """Clean up environment resources"""
        pass


if __name__ == "__main__":
    ChessEnv()
