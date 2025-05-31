from src.piece import Piece

import pygame


class King(Piece):
    def __init__(self, pos, color, board):
        super().__init__(pos, color, board)
        img_path = "resources/assets/" + color[0] + "_king.png"
        self.img = pygame.image.load(img_path)
        self.img = pygame.transform.scale(
            self.img, (board.tile_width - 20, board.tile_height - 20)
        )
        self.notation = "K"

    def get_possible_moves(self, board):
        output = []
        moves = [
            (0, -1),  # north
            (1, -1),  # ne
            (1, 0),  # east
            (1, 1),  # se
            (0, 1),  # south
            (-1, 1),  # sw
            (-1, 0),  # west
            (-1, -1),  # nw
        ]
        for move in moves:
            new_pos = (self.x + move[0], self.y + move[1])
            if (
                new_pos[0] < 8
                and new_pos[0] >= 0
                and new_pos[1] < 8
                and new_pos[1] >= 0
            ):
                output.append([board.get_square_from_pos(new_pos)])
        return output

    def can_castle(self, board):
        """
        Check if castling is possible. Castling is allowed when:
        1. King has not moved
        2. Rook has not moved
        3. No pieces between king and rook
        4. King is not in check
        5. King does not pass through check
        6. King does not end up in check
        """
        if not self.has_moved and not board.is_in_check(self.color):
            if self.color == "white":
                # Queenside castling
                queenside_rook = board.get_piece_from_pos((0, 7))
                if queenside_rook is not None and not queenside_rook.has_moved:
                    # Check if squares between are empty
                    if [board.get_piece_from_pos((i, 7)) for i in range(1, 4)] == [None, None, None]:
                        # Check if king passes through check
                        if (not board.is_in_check(self.color, board_change=[self.pos, (3, 7)]) and
                            not board.is_in_check(self.color, board_change=[self.pos, (2, 7)])):
                            return "queenside"
                
                # Kingside castling
                kingside_rook = board.get_piece_from_pos((7, 7))
                if kingside_rook is not None and not kingside_rook.has_moved:
                    # Check if squares between are empty
                    if [board.get_piece_from_pos((i, 7)) for i in range(5, 7)] == [None, None]:
                        # Check if king passes through check
                        if (not board.is_in_check(self.color, board_change=[self.pos, (5, 7)]) and
                            not board.is_in_check(self.color, board_change=[self.pos, (6, 7)])):
                            return "kingside"
            
            elif self.color == "black":
                # Queenside castling
                queenside_rook = board.get_piece_from_pos((0, 0))
                if queenside_rook is not None and not queenside_rook.has_moved:
                    # Check if squares between are empty
                    if [board.get_piece_from_pos((i, 0)) for i in range(1, 4)] == [None, None, None]:
                        # Check if king passes through check
                        if (not board.is_in_check(self.color, board_change=[self.pos, (3, 0)]) and
                            not board.is_in_check(self.color, board_change=[self.pos, (2, 0)])):
                            return "queenside"
                
                # Kingside castling
                kingside_rook = board.get_piece_from_pos((7, 0))
                if kingside_rook is not None and not kingside_rook.has_moved:
                    # Check if squares between are empty
                    if [board.get_piece_from_pos((i, 0)) for i in range(5, 7)] == [None, None]:
                        # Check if king passes through check
                        if (not board.is_in_check(self.color, board_change=[self.pos, (5, 0)]) and
                            not board.is_in_check(self.color, board_change=[self.pos, (6, 0)])):
                            return "kingside"
        return None

    def get_valid_moves(self, board):
        output = []
        for square in self.get_moves(board):
            if not board.is_in_check(self.color, board_change=[self.pos, square.pos]):
                output.append(square)
        
        # Add castling moves if valid
        castle_side = self.can_castle(board)
        if castle_side == "queenside":
            output.append(board.get_square_from_pos((self.x - 2, self.y)))
        elif castle_side == "kingside":
            output.append(board.get_square_from_pos((self.x + 2, self.y)))
        
        return output
