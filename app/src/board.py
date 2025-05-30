# /* Board.py
import random
import chess
import numpy as np
from src.square import Square
from src.pieces.rook import Rook
from src.pieces.bishop import Bishop
from src.pieces.knight import Knight
from src.pieces.queen import Queen
from src.pieces.king import King
from src.pieces.pawn import Pawn


# Game state checker
class Board:
    def __init__(self, width, height, agent="random", agent_color="black"):
        self.width = width
        self.height = height
        self.tile_width = width // 8
        self.tile_height = height // 8
        self.selected_piece = None
        self.turn = "white"
        self.config = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"],
        ]
        self.equivalences = {
            "r": "bR",
            "n": "bN",
            "b": "bB",
            "q": "bQ",
            "k": "bK",
            "p": "bP",
            "R": "wR",
            "N": "wN",
            "B": "wB",
            "Q": "wQ",
            "K": "wK",
            "P": "wP",
        }
        self.squares = self.generate_squares()
        self.board = None
        self.setup_board()
        self.agent_color = agent_color
        self.agent = agent
        if not isinstance(self.agent, str):
            try:
                self.agent.initialize()
            except AttributeError:
                print("Agent has no initialize method")

    def generate_squares(self):
        output = []
        for y in range(8):
            for x in range(8):
                output.append(Square(x, y, self.tile_width, self.tile_height))
        return output

    def get_square_from_pos(self, pos):
        for square in self.squares:
            if (square.x, square.y) == (pos[0], pos[1]):
                return square

    def get_piece_from_pos(self, pos):
        return self.get_square_from_pos(pos).occupying_piece

    def setup_board(self):
        # Create a Chess board object
        self.board = chess.Board()
        for y, row in enumerate(self.config):
            for x, piece in enumerate(row):
                if piece != "":
                    square = self.get_square_from_pos((x, y))
                    # looking inside contents, what piece does it have
                    if piece[1] == "R":
                        square.occupying_piece = Rook(
                            (x, y), "white" if piece[0] == "w" else "black", self
                        )
                    # as you notice above, we put `self` as argument, or means our class Board
                    elif piece[1] == "N":
                        square.occupying_piece = Knight(
                            (x, y), "white" if piece[0] == "w" else "black", self
                        )
                    elif piece[1] == "B":
                        square.occupying_piece = Bishop(
                            (x, y), "white" if piece[0] == "w" else "black", self
                        )
                    elif piece[1] == "Q":
                        square.occupying_piece = Queen(
                            (x, y), "white" if piece[0] == "w" else "black", self
                        )
                    elif piece[1] == "K":
                        square.occupying_piece = King(
                            (x, y), "white" if piece[0] == "w" else "black", self
                        )
                    elif piece[1] == "P":
                        square.occupying_piece = Pawn(
                            (x, y), "white" if piece[0] == "w" else "black", self
                        )

    def _get_board_state(self):
        """Convert current board state to the format expected by DeepQAgent"""
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

    def _create_simple_env_wrapper(self):
        """Create a simple environment wrapper for the agent"""

        class SimpleEnvWrapper:
            def __init__(self, board):
                self.board = board
                # Create all possible moves mapping (same as ChessEnv)
                self.all_possibles_moves_list = self._create_moves_mapping()
                self.inversed_all_possibles_moves_list = {
                    v: k for k, v in self.all_possibles_moves_list.items()
                }

            def _create_moves_mapping(self):
                """Create mapping of all possible chess moves (same as ChessEnv)"""
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

            @property
            def available_moves(self):
                return list(self.board.legal_moves)

            def move_str_to_index(self, move_str):
                return self.all_possibles_moves_list.get(move_str, 0)

            def move_index_to_str(self, move_index):
                return self.inversed_all_possibles_moves_list.get(move_index, "a1a1")

        return SimpleEnvWrapper(self.board)

    def handle_click(self, mx, my):
        x = mx // self.tile_width
        y = my // self.tile_height
        clicked_square = self.get_square_from_pos((x, y))
        clicked_square_coord = clicked_square.get_coord()

        if self.selected_piece is None:
            if clicked_square.occupying_piece is not None:
                if clicked_square.occupying_piece.color == self.turn:
                    self.selected_piece = clicked_square.occupying_piece
        else:
            selected_piece_coord = self.selected_piece.get_coord()

            # If clicking on the same square, deselect
            if selected_piece_coord == clicked_square_coord:
                self.selected_piece = None
                return

            # Try to move the selected piece
            if self.selected_piece.move(self, clicked_square):
                self.turn = "white" if self.turn == "black" else "black"
                move = chess.Move.from_uci(
                    f"{selected_piece_coord}{clicked_square_coord}"
                )
                self.board.push(move)
                self.selected_piece = None
            # If clicking on another piece of the same color, select it instead
            elif clicked_square.occupying_piece is not None:
                if clicked_square.occupying_piece.color == self.turn:
                    self.selected_piece = clicked_square.occupying_piece
                else:
                    # Trying to capture - this should be handled by the piece.move() method
                    self.selected_piece = None
            else:
                # Clicked on empty square but move was invalid
                self.selected_piece = None

    def agent_move(self):
        # get a random move :
        legal_moves = list(self.board.legal_moves)
        if self.agent_color == "white":
            white_moves = [
                move
                for move in legal_moves
                if self.board.piece_at(move.from_square).color == chess.WHITE
            ]
            if self.agent == "random":
                move = random.choice(white_moves)
            else:
                # Get board state and environment for DeepQAgent
                state = self._get_board_state()
                env = self._create_simple_env_wrapper()
                turn = 0  # 0 for white, 1 for black
                move_str, _ = self.agent.predict(state, env, turn)
                move = chess.Move.from_uci(move_str)
        else:
            black_moves = [
                move
                for move in legal_moves
                if self.board.piece_at(move.from_square).color == chess.BLACK
            ]

            if self.agent == "random":
                move = random.choice(black_moves)
            else:
                # Get board state and environment for DeepQAgent
                state = self._get_board_state()
                env = self._create_simple_env_wrapper()
                turn = 1  # 0 for white, 1 for black
                move_str, _ = self.agent.predict(state, env, turn)
                move = chess.Move.from_uci(move_str)

        self.board.push(move)
        self.turn = "white" if self.turn == "black" else "black"
        self.selected_piece = None

        piece = self.get_piece_from_pos(
            (move.from_square % 8, 7 - move.from_square // 8)
        )
        # get the square from the move
        square = self.get_square_from_pos((move.to_square % 8, 7 - move.to_square // 8))
        # move the piece to the square
        square.occupying_piece = piece
        # remove the piece from the old square
        # piece.square.occupying_piece = None
        # update the piece square
        piece.square = square

        square = self.get_square_from_pos(
            (move.from_square % 8, 7 - move.from_square // 8)
        )
        square.occupying_piece = None

    def is_in_check(
        self, color, board_change=None
    ):  # board_change = [(x1, y1), (x2, y2)]
        output = False
        king_pos = None
        changing_piece = None
        old_square = None
        new_square = None
        new_square_old_piece = None
        if board_change is not None:
            for square in self.squares:
                if square.pos == board_change[0]:
                    changing_piece = square.occupying_piece
                    old_square = square
                    old_square.occupying_piece = None
            for square in self.squares:
                if square.pos == board_change[1]:
                    new_square = square
                    new_square_old_piece = new_square.occupying_piece
                    new_square.occupying_piece = changing_piece
        pieces = [
            i.occupying_piece for i in self.squares if i.occupying_piece is not None
        ]
        if changing_piece is not None:
            if changing_piece.notation == "K":
                king_pos = new_square.pos
        if king_pos == None:
            for piece in pieces:
                if piece.notation == "K" and piece.color == color:
                    king_pos = piece.pos
        for piece in pieces:
            if piece.color != color:
                for square in piece.attacking_squares(self):
                    if square.pos == king_pos:
                        output = True
        if board_change is not None:
            old_square.occupying_piece = changing_piece
            new_square.occupying_piece = new_square_old_piece
        return output

    def is_in_checkmate(self, color):
        output = False
        king = None
        for piece in [i.occupying_piece for i in self.squares]:
            if piece != None:
                if piece.notation == "K" and piece.color == color:
                    king = piece
                    break

        if king is not None and king.get_valid_moves(self) == []:
            if self.is_in_check(color):
                output = True
        return output

    def draw(self, display):
        if self.selected_piece is not None:
            self.get_square_from_pos(self.selected_piece.pos).highlight = True
            for square in self.selected_piece.get_valid_moves(self):
                square.highlight = True
        for square in self.squares:
            square.draw(display)
