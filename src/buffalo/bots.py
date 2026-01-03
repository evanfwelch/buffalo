import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch

from buffalo.encoders import BoardStateEncoder
from buffalo.models import BuffaloQNetwork

from .board import Board, Move, Player


class Bot(ABC):
    def __init__(self, board: Board, player: Player) -> None:
        self.board = board
        self.player = player

    def generate_legal_moves(self) -> List[Move]:
        """Generate all legal moves for this bot's player."""

        assert (
            self.board.current_player == self.player
        ), f"Bot can only generate moves on its turn, but got {self.board.current_player=}"
        return self.board.legal_moves()

    @abstractmethod
    def choose_move(
        self,
    ) -> Optional[Move]:
        """Select a legal move for the bot's player without mutating the board."""
        raise NotImplementedError

    def _choose_random_legal_move(
        self,
    ) -> Optional[Move]:
        legal_moves = self.generate_legal_moves()
        if not legal_moves:
            return None
        return random.choice(legal_moves)


class NaiveBuffalo(Bot):
    def __init__(self, board: Board) -> None:
        super().__init__(board, Player.BUFFALO)

    def choose_move(
        self,
    ) -> Optional[Move]:
        return self._choose_random_legal_move()


class NaiveHunter(Bot):
    def __init__(self, board: Board) -> None:
        super().__init__(board, Player.HUNTERS)

    def choose_move(
        self,
    ) -> Optional[Move]:

        return self._choose_random_legal_move()


class TorchBuffalo(Bot):
    def __init__(
        self,
        board: Board,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(board, Player.BUFFALO)
        self.board_state_encoder = BoardStateEncoder()

        state_size = self.board_state_encoder.state_size
        action_size = self.board_state_encoder.buffalo_action_size
        self.dqn = BuffaloQNetwork(state_size, action_size)
        torch_device = torch.device(device) if device else None
        if model_path is not None:
            print(f"Loading model from {model_path}")
            payload = torch.load(model_path, map_location=torch_device or "cpu")
            self.dqn.load_state_dict(payload["state_dict"])
        if torch_device:
            self.dqn.to(torch_device)

    def load_model(self, model_path: str, device: Optional[str] = None) -> None:
        torch_device = torch.device(device) if device else None
        payload = torch.load(model_path, map_location=torch_device or "cpu")
        self.dqn.load_state_dict(payload["state_dict"])
        if torch_device:
            self.dqn.to(torch_device)

    def choose_move(
        self,
    ) -> Optional[Move]:

        legal_moves = self.generate_legal_moves()
        encoded_state = self.board_state_encoder.encode(self.board)
        encoded_actions = self.board_state_encoder.buffalo_move_one_hot_encoder(legal_moves)
        encoded_state_action_space = self.board_state_encoder.buffalo_joint_state_action_encoder(
            encoded_state,
            encoded_actions,
        )

        assert len(legal_moves) == encoded_state_action_space.shape[0]

        q_hat = self.dqn(encoded_state_action_space)

        assert q_hat.shape[0] == len(legal_moves)

        chosen_move_idx = torch.argmax(q_hat).item()

        return legal_moves[chosen_move_idx]

    def encode_board_state(self) -> torch.Tensor:
        return self.board_state_encoder.encode(self.board)

    def encode_state_action_(self) -> torch.Tensor:
        legal_moves = self.generate_legal_moves()

        encoded_legal_moves = self.board_state_encoder.buffalo_move_one_hot_encoder(legal_moves)
        encoded_board_state = self.encode_board_state()

        return self.board_state_encoder.buffalo_joint_state_action_encoder(
            encoded_board_state,
            encoded_legal_moves,
        )


class TrainedTorchBuffalo(TorchBuffalo):
    def __init__(
        self,
        board: Board,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(board, model_path="trained_models", device="cpu")
