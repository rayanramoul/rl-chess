from loguru import logger
import chess
import gym
import numpy as np
from gym import spaces

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
    def __init__(self):
        super(ChessEnv, self).__init__()
        # Define the observation space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(64,)
        )  # 8x8 board representation

        # Define the action space
        self.action_space = spaces.Discrete(4096)  # 64*64 possible moves

        # Create a Chess board object
        self.board = chess.Board()

        self.all_possibles_moves_list = self.all_possible_moves()
        logger.debug(f"Total possible moves: {self.all_possibles_moves_list}")

    @property
    def number_of_possible_moves(self):
        return len(self.all_possibles_moves_list)

    def reset(self):
        # Reset the board to the initial state
        self.board.reset()

        return self.translate_board()

    @property
    def state(self):
        return self.translate_board()

    def move_str_to_index(self, move):
        logger.debug(f"Move: {dir(move)}")
        logger.debug(f"Move string : {move}")
        return self.all_possibles_moves_list[str(move)]

    def move_index_to_str(self, move):
        logger.debug(f"Move index: {move}")
        if isinstance(move, tuple):
            move = move[1]
        return list(self.all_possibles_moves_list.keys())[move]

    # def step(self, move):
    #     # Make the move on the board
    #     try:
    #         self.board.push(move)
    #         reward = self._get_reward()
    #     except AssertionError:
    #         # Illegal move
    #         self.board.pop()
    #         reward = -10
    #     # Get the reward and check if the game is over
    #     done = self.board.is_game_over()
    #     state = self.translate_board()
    #     # print("Step state shape : ", state.shape)
    #     return state, reward, done, {}
    #
    def step(self, action):
        logger.debug(f"Action: {action}")

        self.board.push(chess.Move.from_uci(action))
        next_state = self.translate_board()
        reward = self._get_reward()
        done = self.board.is_game_over()
        return next_state, reward, done

    def render(self, mode="human"):
        pass

    # def _get_observation(self):
    #     # Convert the board position to a binary observation
    #     observation = []
    #
    #     for square in chess.SQUARES:
    #         piece = self.board.piece_at(square)
    #         if piece is not None:
    #             observation.append(1)  # Piece present
    #             observation.extend(self._encode_piece(piece))
    #         else:
    #             observation.append(0)  # Empty square
    #
    #     return observation

    def _get_reward(self):
        # Simple reward function example
        if self.board.is_checkmate():
            return 1  # Win
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return -1  # Draw
        else:
            return -1  # No reward

    def _action_to_move(self, action):
        # Convert action index to a move
        moves = list(self.board.legal_moves)
        move = moves[action]
        return move

    @property
    def available_moves(self):
        available_moves = list(self.board.legal_moves)
        logger.debug(f"Available moves: {available_moves}")
        return available_moves

    def all_possible_moves(self):
        self.board_all_moves = []
        letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
        for x in letters:
            for y in range(1, 9):
                for x1 in letters:
                    for y1 in range(1, 9):
                        if x == x1 and y == y1:
                            continue
                        self.board_all_moves.append(f"{x}{y}{x1}{y1}")
        # add the promotion moves
        for x in letters:
            for y in [8, 1]:
                for p in ["q", "r", "b", "n"]:
                    diff_y = 1 if y == 1 else -1
                    self.board_all_moves.append(f"{x}{y}{x}{y+diff_y}{p}")

        # Save the legal moves in a txt file
        with open("legal_moves.txt", "w") as f:
            for item in self.board_all_moves:
                f.write("%s\n" % item)

        # create dictionary of all possible moves with unique index
        self.board_all_moves = {move: i for i, move in enumerate(self.board_all_moves)}
        return self.board_all_moves

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
