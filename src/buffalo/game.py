"""Core game loop and move history for Buffalo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Tuple

from .board import Board, Move, PieceType, Player, Position, MoveResult, MoveRecord


class PlayerController(Protocol):
    def choose_move(self, game: "Game") -> Optional[Move]:
        """Return a legal move for the current player, or None if no move."""


class Game:
    """Manages turn-taking, legal moves, and game history.

    Key concept: the Game object asks players for their move, then applies it.

    The player controller could be a bot or a human clicking.
    """

    def __init__(
        self,
        buffalo_controller: Optional[PlayerController] = None,
        hunter_controller: Optional[PlayerController] = None,
        board: Optional[Board] = None,
    ) -> None:
        self.board = board or Board()
        self.buffalo_controller = buffalo_controller
        self.hunter_controller = hunter_controller

        assert hasattr(self.buffalo_controller, "choose_move")
        assert hasattr(self.hunter_controller, "choose_move")

        self.history: List[MoveRecord] = []
        self.move_number = 0
        self.game_over = False
        self.winner: Optional[Player] = None
        self.game_over_reason = ""

    def controller_for_current_player(self) -> Optional[PlayerController]:
        if self.board.current_player == Player.BUFFALO:
            return self.buffalo_controller
        return self.hunter_controller

    def step(self) -> Tuple[MoveResult, Optional[MoveRecord]]:
        """Advance the game using the configured controller for the current player."""

        if self.game_over:
            return None

        controller = self.controller_for_current_player()

        move = controller.choose_move(self)

        assert move is not None, "Controller must choose a valid move."

        move_result, move_record = self.board.move_piece(move.start.x, move.start.y, move.end.x, move.end.y)
        self.history.append(move_record)
        return move_result, move_record
