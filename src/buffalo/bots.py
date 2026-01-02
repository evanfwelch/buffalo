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

    def _choose_random_legal_move(self, game: Optional[Any] = None) -> Optional[Move]:
        legal_moves = self.generate_legal_moves()
        if not legal_moves:
            return None
        return random.choice(legal_moves)


class NaiveBuffalo(Bot):
    def __init__(self, board: Board) -> None:
        super().__init__(board, Player.BUFFALO)

    def choose_move(self, game: Optional[Any] = None) -> Optional[Move]:
        return self._choose_random_legal_move()

class NaiveHunter(Bot):
    def __init__(self, board: Board) -> None:
        super().__init__(board, Player.HUNTERS)

    def choose_move(self, game: Optional[Any] = None) -> Optional[Move]:
        return self._choose_random_legal_move()