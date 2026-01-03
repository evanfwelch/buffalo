"""Encoders for board states and action spaces."""

from __future__ import annotations

from typing import List, Sequence

import torch

from .board import Board, Move, PieceType, Player


class BoardStateEncoder:
    """Encode a :class:`Board` into a flattened one-hot tensor.

    Each board square is represented by a one-hot encoding of the piece type
    occupying that square. The encoding ignores empty squares and flattens the
    representation into a single vector of length
    ``board.width * board.height * num_piece_types``.
    """

    def __init__(self, board_width: int = 11, board_height: int = 7):
        self.board_width = board_width
        self.board_height = board_height
        self.piece_types: Sequence[PieceType] = [
            PieceType.BUFFALO,
            PieceType.DOG,
            PieceType.CHIEF,
        ]
        # +1 for player turn state
        self.state_size = board_width * board_height * len(self.piece_types) + 1

    def encode(self, board: Board) -> torch.Tensor:
        """Return a one-hot encoded representation of ``board``."""

        state = torch.zeros(self.state_size, dtype=torch.float32)
        for (x, y), piece in board.pieces.items():
            square_index = y * self.board_width + x
            type_index = self.piece_types.index(piece.type)
            index = square_index * len(self.piece_types) + type_index
            state[index] = 1.0
        state[-1] = 1.0 if board.current_player == Player.HUNTERS else 0.0
        return state

    def buffalo_move_one_hot_encoder(self, legal_buffalo_moves: List[Move]) -> torch.Tensor:
        """Return a one-hot vector of legal buffalo files."""

        n_moves: int = len(legal_buffalo_moves)

        move_tensor = torch.zeros((n_moves, self.board_width), dtype=torch.float32)
        for i, move in enumerate(legal_buffalo_moves):
            assert move.start.x == move.end.x, "Buffalo moves must be vertical"
            move_tensor[i, move.start.x] = 1.0
        return move_tensor

    def buffalo_joint_state_action_encoder(
        self,
        encoded_board_state: torch.Tensor,
        encoded_buffalo_moves: torch.Tensor,
    ) -> torch.Tensor:
        """Return a joint representation of the buffalo's state and action space."""

        assert len(encoded_board_state.shape) == 1

        n_moves = encoded_buffalo_moves.size(0)
        expanded_state = encoded_board_state.unsqueeze(0).expand(n_moves, -1)
        return torch.cat([expanded_state, encoded_buffalo_moves], dim=1)

    @property
    def buffalo_action_size(self) -> int:
        return self.board_width

    @property
    def chief_action_size(self) -> int:
        return 8

    @property
    def joint_dog_action_size(self) -> int:
        return (self.board_height * self.board_width) * 2
