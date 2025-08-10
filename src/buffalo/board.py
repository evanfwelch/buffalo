from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pygame

class Player(Enum):
    BUFFALO = 0
    HUNTERS = 1

class PieceType(Enum):
    BUFFALO = "B"
    DOG = "D"
    CHIEF = "C"

@dataclass
class Position:
    x: int
    y: int

@dataclass
class Piece:
    type: PieceType
    player: Player

class Board:
    width: int = 11
    height: int = 7

    _INITIAL_DOG_FILES = [3,4,6,7]
    _INITIAL_KING_FILE = 5

    def __init__(self):
        self.pieces: Dict[Tuple[int, int], Piece] = {}
        self.current_player = Player.BUFFALO
        self.initialize_board()

    def initialize_board(self) -> None:
        # Place buffalo pieces on top rank
        for x in range(self.width):
            self.pieces[(x, 0)] = Piece(PieceType.BUFFALO, Player.BUFFALO)

        # Place dogs on 2nd bottom rank
        for x in self._INITIAL_DOG_FILES:
            self.pieces[(x, self.height - 2)] = Piece(PieceType.DOG, Player.HUNTERS)

        # Place chief at center bottom
        self.pieces[(self._INITIAL_KING_FILE, self.height - 2)] = Piece(PieceType.CHIEF, Player.HUNTERS)

    def get_piece_at(self, x: int, y: int) -> Optional[Piece]:
        return self.pieces.get((x, y))

    def switch_player(self):
        self.current_player = Player.HUNTERS if self.current_player == Player.BUFFALO else Player.BUFFALO

    def _is_destination_inside_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_valid_move(self, piece: Piece, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        assert piece.player == self.current_player, "Piece does not belong to the current player"


        if not self._is_destination_inside_board(to_x, to_y):
            return False

        is_destination_empty = self.get_piece_at(to_x, to_y) is None
        destination_not_on_bottom_row = to_y != self.height - 1

        if piece.type == PieceType.BUFFALO:
            is_one_move_down = (to_y == from_y + 1) and (from_x == to_x)
            
            return is_one_move_down and is_destination_empty
        
        if piece.type == PieceType.CHIEF:
            delta_x, delta_y = abs(to_x - from_x), abs(to_y - from_y)
            is_kinglike_move = (delta_x <= 1 and delta_y <= 1)
            piece_at_destination = self.get_piece_at(to_x, to_y)

            return is_kinglike_move and (is_destination_empty or (piece_at_destination and piece_at_destination.player != piece.player)) and destination_not_on_bottom_row
        
        if piece.type == PieceType.DOG:
            delta_x, delta_y = abs(to_x - from_x), abs(to_y - from_y)
            is_queen_like_move = (delta_x == delta_y) or (to_x == from_x) or (to_y == from_y)

            no_pieces_between = True
            if is_queen_like_move:
                step_x = 1 if to_x > from_x else -1 if to_x < from_x else 0
                step_y = 1 if to_y > from_y else -1 if to_y < from_y else 0
                curr_x, curr_y = from_x + step_x, from_y + step_y
                while (curr_x, curr_y) != (to_x, to_y):
                    if self.get_piece_at(curr_x, curr_y) is not None:
                        no_pieces_between = False
                        break
                    curr_x += step_x
                    curr_y += step_y

            return is_queen_like_move and is_destination_empty and no_pieces_between and destination_not_on_bottom_row

        return False

    def check_for_winner(self) -> Optional[Player]:
        # any buffalo on bottom row
        for x in range(self.width):
            piece = self.get_piece_at(x, self.height - 1)
            if piece is not None:
                if piece.type == PieceType.BUFFALO:
                    return Player.BUFFALO

        return None

    def move_piece(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        piece = self.get_piece_at(from_x, from_y)
        if not piece:
            return False
        
        if piece.player != self.current_player:
            return False

        if not self._is_valid_move(piece, from_x, from_y, to_x, to_y):
            return False

        self.pieces[(to_x, to_y)] = piece
        del self.pieces[(from_x, from_y)]

        # Switch current player
        self.switch_player()
        return True
