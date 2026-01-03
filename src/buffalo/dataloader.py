"""Dataset utilities for training from saved game CSV files."""

from __future__ import annotations

from pathlib import Path
import csv
from typing import Iterable, Iterator, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

from .board import Board, Player
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
