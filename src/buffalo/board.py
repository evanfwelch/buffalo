from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

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
