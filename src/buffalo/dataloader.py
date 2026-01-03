"""Dataset utilities for training from saved game CSV files."""

from __future__ import annotations

from pathlib import Path
import csv
from typing import Iterable, Iterator, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

from .board import Board, PieceType, Player
from .encoders import BoardStateEncoder


def _iter_csv_files(root: Path) -> Iterable[Path]:
    if root.is_dir():
        yield from sorted(root.glob("*.csv"))
    else:
        yield root


def _parse_player(value: str) -> Optional[Player]:
    if not value:
        return None
    if "BUFFALO" in value:
        return Player.BUFFALO
    if "HUNTERS" in value:
        return Player.HUNTERS
    return None


def _captured_buffalo(value: str) -> bool:
    if not value or value == "None":
        return False
    return "PieceType.BUFFALO" in value


def _captured_piece_type(value: str) -> Optional[PieceType]:
    if not value or value == "None":
        return None
    if "PieceType.BUFFALO" in value:
        return PieceType.BUFFALO
    if "PieceType.DOG" in value:
        return PieceType.DOG
    if "PieceType.CHIEF" in value:
        return PieceType.CHIEF
    return None


def _parse_piece_type(value: str) -> Optional[PieceType]:
    if not value or value == "None":
        return None
    if "PieceType.BUFFALO" in value:
        return PieceType.BUFFALO
    if "PieceType.DOG" in value:
        return PieceType.DOG
    if "PieceType.CHIEF" in value:
        return PieceType.CHIEF
    return None


def _encode_chief_direction(from_x: int, from_y: int, to_x: int, to_y: int) -> torch.Tensor:
    direction = torch.zeros(8, dtype=torch.float32)
    dx = to_x - from_x
    dy = to_y - from_y
    mapping = {
        (0, -1): 0,  # up
        (0, 1): 1,  # down
        (-1, 0): 2,  # left
        (1, 0): 3,  # right
        (1, -1): 4,  # up-right
        (-1, -1): 5,  # up-left
        (1, 1): 6,  # down-right
        (-1, 1): 7,  # down-left
    }
    index = mapping.get((dx, dy))
    if index is not None:
        direction[index] = 1.0
    return direction


def _encode_dog_positions(board: Board) -> torch.Tensor:
    flat = torch.zeros(board.width * board.height, dtype=torch.float32)
    for (x, y), piece in board.pieces.items():
        if piece.type == PieceType.DOG:
            flat[y * board.width + x] = 1.0
    return flat


class BuffaloGameDataset(IterableDataset):
    """Iterate over (state, action, reward, next_state) transitions from CSV logs.

    The dataset yields transitions for buffalo turns. The state is the board before
    a buffalo move, and the next state is the board after the hunters' subsequent move.
    """

    def __init__(
        self,
        root: str | Path,
        encoder: Optional[BoardStateEncoder] = None,
        win_reward: float = 1.0,
        loss_reward: float = -1.0,
        capture_penalty: float = -0.1,
    ) -> None:
        self.root = Path(root)
        self.encoder = encoder or BoardStateEncoder()
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.capture_penalty = capture_penalty

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]:
        for path in _iter_csv_files(self.root):
            yield from self._iter_file(path)

    def _iter_file(self, path: Path) -> Iterator[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            previous_row: Optional[dict] = None
            for row in reader:
                if previous_row is None:
                    previous_row = row
                    continue

                prev_player = _parse_player(previous_row.get("player", ""))
                curr_player = _parse_player(row.get("player", ""))
                if prev_player != Player.BUFFALO or curr_player != Player.HUNTERS:
                    previous_row = row
                    continue

                board_before = Board.deserialize(previous_row["board_before"])
                board_after = Board.deserialize(row["board_after"])

                state = self.encoder.encode(board_before)
                next_state = self.encoder.encode(board_after)

                action = torch.zeros(self.encoder.board_width, dtype=torch.float32)
                action[int(previous_row["from_x"])] = 1.0

                reward = 0.0
                winner = _parse_player(row.get("winner_after_move", ""))
                if winner == Player.BUFFALO:
                    reward = self.win_reward
                elif winner == Player.HUNTERS:
                    reward = self.loss_reward
                if _captured_buffalo(row.get("captured_piece", "")):
                    reward += self.capture_penalty

                yield state, action, reward, next_state
                previous_row = row


class HunterGameDataset(IterableDataset):
    """Iterate over (state, action, reward, next_state) transitions for hunters.

    The dataset yields transitions for hunter turns. The state is the board before
    a hunter move, and the next state is the board after the buffalo's subsequent move.
    """

    def __init__(
        self,
        root: str | Path,
        encoder: Optional[BoardStateEncoder] = None,
        win_reward: float = 1.0,
        loss_reward: float = -1.0,
        capture_reward: float = 0.1,
    ) -> None:
        self.root = Path(root)
        self.encoder = encoder or BoardStateEncoder()
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.capture_reward = capture_reward

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]:
        for path in _iter_csv_files(self.root):
            yield from self._iter_file(path)

    def _iter_file(self, path: Path) -> Iterator[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            previous_row: Optional[dict] = None
            for row in reader:
                if previous_row is None:
                    previous_row = row
                    continue

                prev_player = _parse_player(previous_row.get("player", ""))
                curr_player = _parse_player(row.get("player", ""))
                if prev_player != Player.HUNTERS or curr_player != Player.BUFFALO:
                    previous_row = row
                    continue

                board_before = Board.deserialize(previous_row["board_before"])
                board_after = Board.deserialize(row["board_after"])

                state = self.encoder.encode(board_before)
                next_state = self.encoder.encode(board_after)

                piece_type = _parse_piece_type(previous_row.get("piece_type", ""))
                from_x = int(previous_row["from_x"])
                from_y = int(previous_row["from_y"])
                to_x = int(previous_row["to_x"])
                to_y = int(previous_row["to_y"])

                chief_action = torch.zeros(8, dtype=torch.float32)
                if piece_type == PieceType.CHIEF:
                    chief_action = _encode_chief_direction(from_x, from_y, to_x, to_y)

                dogs_before = _encode_dog_positions(board_before)
                dogs_after = _encode_dog_positions(board_after)
                action = torch.cat([chief_action, dogs_before, dogs_after], dim=0)

                reward = 0.0
                winner = _parse_player(row.get("winner_after_move", ""))
                if winner == Player.HUNTERS:
                    reward = self.win_reward
                elif winner == Player.BUFFALO:
                    reward = self.loss_reward

                captured_piece = _captured_piece_type(row.get("captured_piece", ""))
                if captured_piece == PieceType.BUFFALO:
                    reward += self.capture_reward

                yield state, action, reward, next_state
                previous_row = row
