"""Dataset utilities for training from saved game JSONL files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple
from abc import ABC, abstractmethod

import torch
from torch.utils.data import IterableDataset

from .board import Board, Move, MoveRecord, PieceType, Player
from .encoders import BoardStateEncoder


def _iter_jsonl_files(root: Path) -> Iterable[Path]:
    if root.is_dir():
        yield from sorted(root.glob("*.jsonl"))
    else:
        yield root


class BaseGameDataset(IterableDataset, ABC):
    """Base dataset for extracting transitions from JSONL game logs."""

    def __init__(
        self,
        root: str | Path,
        encoder: Optional[BoardStateEncoder] = None,
        prev_player: Player = Player.BUFFALO,
        curr_player: Player = Player.HUNTERS,
        win_reward: float = 1.0,
        loss_reward: float = -1.0,
        capture_delta: float = 0.0,
        capture_piece: Optional[PieceType] = None,
    ) -> None:
        self.root = Path(root)
        self.encoder = encoder or BoardStateEncoder()
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.capture_delta = capture_delta
        self.capture_piece = capture_piece
        self.prev_player = prev_player
        self.curr_player = curr_player

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]:
        for path in _iter_jsonl_files(self.root):
            previous_row: Optional[MoveRecord] = None
            for record in self._iter_records(path):
                if previous_row is None:
                    previous_row = record
                    continue

                if (
                    previous_row.player != self.prev_player
                    or record.player != self.curr_player
                ):
                    previous_row = record
                    continue

                board_before = Board.from_pieces(previous_row.pieces_before, previous_row.player)
                next_player = Player.HUNTERS if record.player == Player.BUFFALO else Player.BUFFALO
                board_after = Board.from_pieces(record.pieces_after, next_player)

                state = self.encoder.encode(board_before)
                next_state = self.encoder.encode(board_after)
                action = self._encode_action(previous_row, board_before, board_after)

                if action is None:
                    previous_row = record
                    continue

                reward = self._reward_from_row(record)

                yield state, action, reward, next_state
                previous_row = record

    def _iter_records(self, path: Path) -> Iterator[MoveRecord]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield MoveRecord.from_json(line)

    def _reward_from_row(self, row: MoveRecord) -> float:
        reward = 0.0
        if row.winner_after_move == self.prev_player:
            reward = self.win_reward
        elif row.winner_after_move == self.curr_player:
            reward = self.loss_reward

        if self.capture_piece is not None and self.capture_delta != 0.0:
            captured_piece = row.captured_piece
            if captured_piece == self.capture_piece:
                reward += self.capture_delta

        return reward

    @abstractmethod
    def _encode_action(
        self,
        previous_row: MoveRecord,
        board_before: Board,
        board_after: Board,
    ) -> Optional[torch.Tensor]:
        raise NotImplementedError


class BuffaloGameDataset(BaseGameDataset):
    """Iterate over (state, action, reward, next_state) transitions from JSONL logs.

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
        super().__init__(
            root=root,
            encoder=encoder,
            win_reward=win_reward,
            loss_reward=loss_reward,
            capture_delta=capture_penalty,
            capture_piece=PieceType.BUFFALO,
            prev_player=Player.BUFFALO,
            curr_player=Player.HUNTERS,
        )

    def _encode_action(
        self,
        previous_row: MoveRecord,
        board_before: Board,
        board_after: Board,
    ) -> Optional[torch.Tensor]:
        from_pos = previous_row.from_pos
        to_pos = previous_row.to_pos
        piece = board_before.get_piece_at(from_pos.x, from_pos.y)
        if piece is None:
            return None
        move = Move(player=Player.BUFFALO, piece=piece, start=from_pos, end=to_pos)
        joint = self.encoder.joint_state_action_encoder(board_before, [move])
        return joint[0, self.encoder.state_size :]


class HunterGameDataset(BaseGameDataset):
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
        super().__init__(
            root=root,
            encoder=encoder,
            win_reward=win_reward,
            loss_reward=loss_reward,
            capture_delta=capture_reward,
            capture_piece=PieceType.BUFFALO,
            prev_player=Player.HUNTERS,
            curr_player=Player.BUFFALO,
        )

    def _encode_action(
        self,
        previous_row: MoveRecord,
        board_before: Board,
        board_after: Board,
    ) -> Optional[torch.Tensor]:
        from_pos = previous_row.from_pos
        to_pos = previous_row.to_pos
        piece = board_before.get_piece_at(from_pos.x, from_pos.y)
        if piece is None:
            return None
        move = Move(player=Player.HUNTERS, piece=piece, start=from_pos, end=to_pos)
        joint = self.encoder.joint_state_action_encoder(board_before, [move])
        return joint[0, self.encoder.state_size :]
