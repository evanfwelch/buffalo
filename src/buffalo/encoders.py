"""Encoders for board states and action spaces."""

from __future__ import annotations

from typing import List, Sequence

import torch

from .board import Board, Move, PieceType, Player, Position


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

    def joint_state_action_encoder(
        self,
        board: Board,
        legal_moves: List[Move],
    ) -> torch.Tensor:
        """Encode a board and its legal moves into joint state-action vectors."""

        encoded_state = self.encode(board)
        player = self._infer_player(board, legal_moves)
        if player == Player.BUFFALO:
            encoded_actions = self.buffalo_move_one_hot_encoder(legal_moves)
            return self.buffalo_joint_state_action_encoder(encoded_state, encoded_actions)

        encoded_actions = self._hunter_move_action_encoder(board, legal_moves)
        return self._hunter_joint_state_action_encoder(encoded_state, encoded_actions)

    def _hunter_joint_state_action_encoder(
        self,
        encoded_board_state: torch.Tensor,
        encoded_hunter_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return a joint representation of the hunters' state and action space."""

        assert encoded_hunter_actions.dim() == 2

        n_moves = encoded_hunter_actions.size(0)
        expanded_state = encoded_board_state.unsqueeze(0).expand(n_moves, -1)
        return torch.cat([expanded_state, encoded_hunter_actions], dim=1)

    def _infer_player(self, board: Board, legal_moves: List[Move]) -> Player:
        if legal_moves:
            return legal_moves[0].player
        return board.current_player

    def _hunter_move_action_encoder(
        self,
        board: Board,
        legal_moves: List[Move],
    ) -> torch.Tensor:
        action_size = 8 + (self.board_width * self.board_height * 2)
        if not legal_moves:
            return torch.zeros((0, action_size), dtype=torch.float32)

        dogs_before = self._encode_dog_positions(board)
        actions: List[torch.Tensor] = []
        for move in legal_moves:
            chief_action = self._encode_chief_direction(move.start, move.end, move.piece.type)
            board_after = self._apply_move(board, move)
            dogs_after = self._encode_dog_positions(board_after)
            actions.append(torch.cat([chief_action, dogs_before, dogs_after], dim=0))
        return torch.stack(actions, dim=0)

    def _encode_dog_positions(self, board: Board) -> torch.Tensor:
        flat = torch.zeros(self.board_width * self.board_height, dtype=torch.float32)
        for (x, y), piece in board.pieces.items():
            if piece.type == PieceType.DOG:
                flat[y * self.board_width + x] = 1.0
        return flat

    def _encode_chief_direction(
        self,
        from_pos: Position,
        to_pos: Position,
        piece_type: PieceType,
    ) -> torch.Tensor:
        direction = torch.zeros(8, dtype=torch.float32)
        if piece_type != PieceType.CHIEF:
            return direction
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y
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

    def _apply_move(self, board: Board, move: Move) -> Board:
        cloned = Board.deserialize(board.serialize())
        cloned.current_player = move.player
        cloned.move_piece(move.start.x, move.start.y, move.end.x, move.end.y, with_record=False)
        return cloned

    @property
    def buffalo_action_size(self) -> int:
        return self.board_width

    @property
    def chief_action_size(self) -> int:
        return 8

    @property
    def joint_dog_action_size(self) -> int:
        return (self.board_height * self.board_width) * 2
