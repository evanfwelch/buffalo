from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


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


@dataclass
class Move:
    player: Player
    piece: Piece
    start: Position
    end: Position


class GameOverReason(Enum):
    # NOTE: AFAIK chief and dogs can never get stalemated/stuck
    BUFFALO_CROSSED = "buffalo_crossed"
    BUFFALO_STUCK = "buffalo_stuck"
    BUFFALO_EXTINCT = "buffalo_extinct"


@dataclass
class MoveResult:
    captured_piece: Optional[Piece]
    winner_after_move: Optional[Player]
    game_over_reason: Optional[GameOverReason]


@dataclass(frozen=True)
class MoveRecord:
    move_number: int
    player: Player
    piece_type: Optional[PieceType]
    from_pos: Optional[Position]
    to_pos: Optional[Position]
    board_before: str
    board_after: str
    move_result: MoveResult


class Board:
    width: int = 11
    height: int = 7

    _INITIAL_DOG_FILES = [3, 4, 6, 7]
    _INITIAL_KING_FILE = 5

    def __init__(self):
        self.pieces: Dict[Tuple[int, int], Piece] = {}
        self.current_player = Player.BUFFALO
        self.initialize_board()
        self.move_number = 0

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

        # a move is not considered valid if it doesn't change position
        if from_x == to_x and from_y == to_y:
            return False

        piece_at_destination = self.get_piece_at(to_x, to_y)
        is_destination_empty = piece_at_destination is None
        destination_not_on_bottom_row = to_y != self.height - 1
        destination_not_on_top_row = to_y != 0

        if piece.type == PieceType.BUFFALO:
            is_one_move_down = (to_y == from_y + 1) and (from_x == to_x)

            return is_one_move_down and is_destination_empty

        if piece.type == PieceType.CHIEF:
            delta_x, delta_y = abs(to_x - from_x), abs(to_y - from_y)
            is_kinglike_move = delta_x <= 1 and delta_y <= 1

            # must be king-like
            if not is_kinglike_move:
                return False

            # King can move to the bottom row
            if not destination_not_on_bottom_row:
                return False

            # king cannot be on top row
            if not destination_not_on_top_row:
                return False

            # King can move to empty square
            if is_destination_empty:
                return True

            if piece_at_destination.player != piece.player:
                return True

            return False

        if piece.type == PieceType.DOG:

            # dogs cannot capture, only block
            if not is_destination_empty:
                return False

            if not destination_not_on_top_row:
                return False

            if not destination_not_on_bottom_row:
                return False

            delta_x, delta_y = abs(to_x - from_x), abs(to_y - from_y)
            is_queen_like_move = (delta_x == delta_y) or (to_x == from_x) or (to_y == from_y)

            if not is_queen_like_move:
                return False

            no_pieces_between = True

            step_x = 1 if to_x > from_x else -1 if to_x < from_x else 0
            step_y = 1 if to_y > from_y else -1 if to_y < from_y else 0
            curr_x, curr_y = from_x + step_x, from_y + step_y
            while (curr_x, curr_y) != (to_x, to_y):
                if self.get_piece_at(curr_x, curr_y) is not None:
                    no_pieces_between = False
                    break
                curr_x += step_x
                curr_y += step_y

            return no_pieces_between

        return False

    def check_for_winner(self) -> Tuple[Optional[Player], GameOverReason]:
        # any buffalo on bottom row
        for x in range(self.width):
            piece = self.get_piece_at(x, self.height - 1)
            if piece is not None:
                if piece.type == PieceType.BUFFALO:
                    return Player.BUFFALO, GameOverReason.BUFFALO_CROSSED

        # if buffalo's turn and no legal moves, hunters win. This includes the case where no more buffalo left to move
        if self.current_player == Player.BUFFALO and not self.legal_moves():
            return Player.HUNTERS, GameOverReason.BUFFALO_STUCK

        # if all buffalo are extinct, chief wins
        any_buffalo_remaining = any(piece.type == PieceType.BUFFALO for piece in self.pieces.values())
        if not any_buffalo_remaining:
            return Player.HUNTERS, GameOverReason.BUFFALO_EXTINCT

        return None, None

    def move_piece(
        self, from_x: int, from_y: int, to_x: int, to_y: int, with_record: bool = True
    ) -> Tuple[MoveResult, Optional[MoveRecord]]:
        piece = self.get_piece_at(from_x, from_y)
        if not piece:
            raise ValueError("No piece at the starting position")

        if piece.player != self.current_player:
            raise ValueError("Piece does not belong to current player")

        if not self._is_valid_move(piece, from_x, from_y, to_x, to_y):
            raise ValueError("Invalid move")

        captured_piece = self.get_piece_at(to_x, to_y)

        if with_record:
            board_before: str = self.serialize()

        self.pieces[(to_x, to_y)] = piece
        del self.pieces[(from_x, from_y)]

        if with_record:
            board_after: str = self.serialize()

        # increment move number
        self.move_number += 1

        # Switch current player
        self.switch_player()

        # check for winner
        winning_player, reason = self.check_for_winner()

        move_result = MoveResult(
            captured_piece=captured_piece,
            winner_after_move=winning_player,
            game_over_reason=reason,
        )

        if with_record:
            move_record = MoveRecord(
                move_number=self.move_number,
                player=piece.player,
                piece_type=piece.type,
                from_pos=Position(from_x, from_y),
                to_pos=Position(to_x, to_y),
                board_before=board_before,
                board_after=board_after,
                move_result=move_result,
            )
            return move_result, move_record
        return move_result, None

    def legal_moves(self) -> List[Move]:
        """Return legal moves for the current player without mutating the board."""

        legal_moves = []
        for (from_x, from_y), piece in self.pieces.items():
            if piece.player != self.current_player:
                continue
            for to_x in range(self.width):
                for to_y in range(self.height):
                    if self._is_valid_move(piece, from_x, from_y, to_x, to_y):
                        move = Move(
                            player=self.current_player,
                            piece=piece,
                            start=Position(from_x, from_y),
                            end=Position(to_x, to_y),
                        )
                        legal_moves.append(move)
        return legal_moves

    def serialize(self) -> str:
        """Serialize the board from top row (y=0) to bottom (y=height-1)."""

        rows: List[str] = []
        for y in range(self.height):
            row: List[str] = []
            for x in range(self.width):
                piece = self.get_piece_at(x, y)
                row.append(piece.type.value if piece else ".")
            rows.append("".join(row))
        return "/".join(rows)

    @classmethod
    def deserialize(cls, data: str) -> "Board":
        """Create a board from serialized rows produced by serialize()."""

        board = cls()
        board.pieces = {}
        board.current_player = Player.BUFFALO

        rows = data.split("/")
        if len(rows) != board.height:
            raise ValueError("Serialized board has an unexpected number of rows.")

        for y, row in enumerate(rows):
            if len(row) != board.width:
                raise ValueError("Serialized board has an unexpected row width.")
            for x, char in enumerate(row):
                if char == ".":
                    continue
                try:
                    piece_type = PieceType(char)
                except ValueError as exc:
                    raise ValueError(f"Unknown piece token: {char}") from exc
                player = Player.BUFFALO if piece_type == PieceType.BUFFALO else Player.HUNTERS
                board.pieces[(x, y)] = Piece(piece_type, player)
        return board
