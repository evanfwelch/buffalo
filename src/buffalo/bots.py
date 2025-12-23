import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from .board import Board, Move, Player


class Bot(ABC):
    def __init__(self, board: Board, player: Player) -> None:
        self.board = board
        self.player = player

    def generate_legal_moves(self) -> List[Move]:
        """Generate all legal moves for this bot's player."""

        assert (
            self.board.current_player == self.player
        ), "Bot can only generate moves on its turn"
        return self.board.legal_moves()

    @abstractmethod
    def choose_move(self, game: Optional[Any] = None) -> Optional[Move]:
        """Select a legal move for the bot's player without mutating the board."""
        raise NotImplementedError

    def make_move(self) -> bool:
        """Make a move for the bot's player. Returns True if a move was made, False otherwise."""
        move = self.choose_move()
        if move is None:
            return False
        from_pos, to_pos = move
        return self.board.move_piece(from_pos.x, from_pos.y, to_pos.x, to_pos.y)

class NaiveBuffalo(Bot):
    def __init__(self, board: Board) -> None:
        super().__init__(board, Player.BUFFALO)

    def choose_move(self, game: Optional[Any] = None) -> Optional[Move]:
        legal_moves = self.generate_legal_moves()
        if not legal_moves:
            return None
        return random.choice(legal_moves)

class NaiveHunter(Bot):
    def __init__(self, board: Board) -> None:
        super().__init__(board, Player.HUNTERS)

    def choose_move(self, game: Optional[Any] = None) -> Optional[Move]:
        legal_moves = self.generate_legal_moves()
        one_step_moves = [
            move
            for move in legal_moves
            if max(
                abs(move.end.x - move.start.x),
                abs(move.end.y - move.start.y),
            )
            == 1
        ]
        if not one_step_moves:
            return None
        return random.choice(one_step_moves)
