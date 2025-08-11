import random
from abc import ABC, abstractmethod
from typing import List, Tuple

from .board import Board, PieceType, Player, Position

class Bot(ABC):
    def __init__(self, board: Board, player: Player) -> None:
        self.board = board
        self.player = player

    def generate_legal_moves(self) -> List[Tuple[Position, Position]]:
        """Generate all legal moves for this bot's player.

        Uses the board's existing move validation logic to enumerate
        moves without mutating the board state.
        """

        assert (
            self.board.current_player == self.player
        ), "Bot can only generate moves on its turn"

        legal_moves: List[Tuple[Position, Position]] = []
        for (from_x, from_y), piece in self.board.pieces.items():
            if piece.player != self.player:
                continue
            for to_x in range(self.board.width):
                for to_y in range(self.board.height):
                    if self.board._is_valid_move(piece, from_x, from_y, to_x, to_y):
                        legal_moves.append(
                            (Position(from_x, from_y), Position(to_x, to_y))
                        )
        return legal_moves

    @abstractmethod
    def make_move(self) -> bool:
        """Make a move for the bot's player. Returns True if a move was made, False otherwise."""
        pass

class NaiveBuffalo(Bot):
    def __init__(self, board: Board) -> None:
        super().__init__(board, Player.BUFFALO)

    def make_move(self) -> bool:
        # Get all buffalo positions
        buffalo_positions = [
            Position(x, y) for (x, y), piece in self.board.pieces.items()
            if piece.type == PieceType.BUFFALO and piece.player == self.player
        ]

        # SHUFFALO
        random.shuffle(buffalo_positions)  # Shuffle to pick randomly

        for position in buffalo_positions:
            from_x, from_y = position.x, position.y
            to_x, to_y = from_x, from_y + 1  # Attempt to move forward
            if self.board.move_piece(from_x, from_y, to_x, to_y):
                return True  # Move successful


        return False  # No valid moves found

class NaiveHunter(Bot):
    def __init__(self, board: Board) -> None:
        super().__init__(board, Player.HUNTERS)

    def make_move(self) -> bool:
        # Get all hunter pieces (dogs and chief) positions
        hunter_positions = [
            Position(x, y) for (x, y), piece in self.board.pieces.items()
            if piece.player == self.player
        ]

        random.shuffle(hunter_positions)  # Shuffle to pick randomly

        for position in hunter_positions:
            from_x, from_y = position.x, position.y
            # Generate all possible moves of length 1
            possible_moves = [
                Position(from_x + dx, from_y + dy)
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                if (dx != 0 or dy != 0)  # Exclude staying in place
            ]

            random.shuffle(possible_moves)  # Shuffle to pick randomly

            for move in possible_moves:
                to_x, to_y = move.x, move.y
                if self.board.move_piece(from_x, from_y, to_x, to_y):
                    return True  # Move successful

        return False  # No valid moves found
