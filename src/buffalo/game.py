"""Core game loop and move history for Buffalo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol

from .board import Board, Move, PieceType, Player, Position


class PlayerController(Protocol):
    def choose_move(self, game: "Game") -> Optional[Move]:
        """Return a legal move for the current player, or None if no move."""


@dataclass(frozen=True)
class MoveRecord:
    move_number: int
    player: Player
    piece_type: Optional[PieceType]
    from_pos: Optional[Position]
    to_pos: Optional[Position]
    board_before: str
    board_after: str
    captured_piece: Optional[PieceType]
    legal_moves: int
    move_made: bool
    game_over: bool
    winner: Optional[Player]
    game_over_reason: str


class Game:
    """Manages turn-taking, legal moves, and game history."""

    def __init__(
        self,
        buffalo_controller: Optional[PlayerController] = None,
        hunter_controller: Optional[PlayerController] = None,
        board: Optional[Board] = None,
    ) -> None:
        self.board = board or Board()
        self.buffalo_controller = buffalo_controller
        self.hunter_controller = hunter_controller
        self.history: List[MoveRecord] = []
        self.move_number = 0
        self.game_over = False
        self.winner: Optional[Player] = None
        self.game_over_reason = ""

    def controller_for_current_player(self) -> Optional[PlayerController]:
        if self.board.current_player == Player.BUFFALO:
            return self.buffalo_controller
        return self.hunter_controller

    def legal_moves(self) -> List[Move]:
        return self.board.legal_moves()

    def step(self) -> Optional[MoveRecord]:
        """Advance the game using the configured controller for the current player."""

        if self.game_over:
            return None

        controller = self.controller_for_current_player()
        if controller is None:
            return None

        legal_moves = self.legal_moves()
        if not legal_moves:
            return self._record_no_moves(legal_moves_count=0)

        move = controller.choose_move(self)
        if move is None:
            return self._record_no_moves(legal_moves_count=len(legal_moves))

        return self.apply_move(move.start, move.end, legal_moves_count=len(legal_moves))

    def apply_move(
        self,
        from_pos: Position,
        to_pos: Position,
        legal_moves_count: Optional[int] = None,
    ) -> Optional[MoveRecord]:
        """Apply a move for the current player and record it if valid."""

        if self.game_over:
            return None

        mover = self.board.current_player
        board_before = serialize_board(self.board)
        legal_moves = self.legal_moves()
        if legal_moves_count is None:
            legal_moves_count = len(legal_moves)

        piece = self.board.get_piece_at(from_pos.x, from_pos.y)
        if piece is None or piece.player != mover:
            return None

        captured_piece = self.board.get_piece_at(to_pos.x, to_pos.y)
        moved = self.board.move_piece(from_pos.x, from_pos.y, to_pos.x, to_pos.y)
        if not moved:
            return None

        board_after = serialize_board(self.board)
        self.move_number += 1

        record = self._build_record(
            move_number=self.move_number,
            player=mover,
            piece_type=piece.type,
            from_pos=from_pos,
            to_pos=to_pos,
            board_before=board_before,
            board_after=board_after,
            captured_piece=captured_piece.type if captured_piece else None,
            legal_moves_count=legal_moves_count,
        )
        self.history.append(record)
        return record

    def _record_no_moves(self, legal_moves_count: int) -> MoveRecord:
        board_state = serialize_board(self.board)
        record = self._build_record(
            move_number=self.move_number,
            player=self.board.current_player,
            piece_type=None,
            from_pos=None,
            to_pos=None,
            board_before=board_state,
            board_after=board_state,
            captured_piece=None,
            legal_moves_count=legal_moves_count,
            move_made=False,
            game_over_reason="no_moves",
        )
        self.history.append(record)
        return record

    def _build_record(
        self,
        move_number: int,
        player: Player,
        piece_type: Optional[PieceType],
        from_pos: Optional[Position],
        to_pos: Optional[Position],
        board_before: str,
        board_after: str,
        captured_piece: Optional[PieceType],
        legal_moves_count: int,
        move_made: bool = True,
        game_over_reason: str = "",
    ) -> MoveRecord:
        winner = self.board.check_for_winner()
        game_over = winner is not None
        if not game_over and move_made:
            if not self.board.legal_moves():
                game_over = True
                game_over_reason = "no_moves"
        if not move_made and game_over_reason:
            game_over = True

        if winner == Player.BUFFALO:
            game_over_reason = "buffalo_reached_end"

        if game_over:
            self.game_over = True
            self.winner = winner
            self.game_over_reason = game_over_reason

        return MoveRecord(
            move_number=move_number,
            player=player,
            piece_type=piece_type,
            from_pos=from_pos,
            to_pos=to_pos,
            board_before=board_before,
            board_after=board_after,
            captured_piece=captured_piece,
            legal_moves=legal_moves_count,
            move_made=move_made,
            game_over=game_over,
            winner=winner,
            game_over_reason=game_over_reason,
        )


def serialize_board(board: Board) -> str:
    """Serialize the board from top row (y=0) to bottom (y=height-1)."""

    rows: List[str] = []
    for y in range(board.height):
        row: List[str] = []
        for x in range(board.width):
            piece = board.get_piece_at(x, y)
            row.append(piece.type.value if piece else ".")
        rows.append("".join(row))
    return "/".join(rows)


def record_to_row(record: MoveRecord) -> dict:
    return {
        "move_number": record.move_number,
        "player": record.player.name,
        "piece_type": record.piece_type.value if record.piece_type else "",
        "from_x": record.from_pos.x if record.from_pos else "",
        "from_y": record.from_pos.y if record.from_pos else "",
        "to_x": record.to_pos.x if record.to_pos else "",
        "to_y": record.to_pos.y if record.to_pos else "",
        "board_before": record.board_before,
        "board_after": record.board_after,
        "captured": bool(record.captured_piece),
        "captured_piece": record.captured_piece.value if record.captured_piece else "",
        "legal_moves": record.legal_moves,
        "move_made": record.move_made,
        "game_over": record.game_over,
        "winner": record.winner.name if record.winner else "",
        "game_over_reason": record.game_over_reason,
    }


def csv_fields() -> Iterable[str]:
    return [
        "move_number",
        "player",
        "piece_type",
        "from_x",
        "from_y",
        "to_x",
        "to_y",
        "board_before",
        "board_after",
        "captured",
        "captured_piece",
        "legal_moves",
        "move_made",
        "game_over",
        "winner",
        "game_over_reason",
    ]
