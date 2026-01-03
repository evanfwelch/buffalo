import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from buffalo.encoders import BoardStateEncoder
import torch

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

    def __init__(self, board: Board) -> None:
        super().__init__(board, Player.BUFFALO)
        self.board_state_encoder = BoardStateEncoder()

        state_size = self.board_state_encoder.state_size
        action_size = self.board_state_encoder.buffalo_action_size

        self.dqn = torch.nn.Sequential(
            torch.nn.Linear(state_size + action_size, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1), torch.nn.Tanh()
        )

    def get_buffalo_input_dim(self) -> int:
        return self.board_state_encoder.state_size + self.board_state_encoder.buffalo_action_size

    def choose_move(
        self,
    ) -> Optional[Move]:

        legal_moves = self.generate_legal_moves()
        encoded_state_action_space = self.board_state_encoder.buffalo_joint_state_action_encoder(
            self.board_state_encoder.encode(self.board),
            self.board_state_encoder.buffalo_move_one_hot_encoder(legal_moves),
        )

        assert len(legal_moves) == encoded_state_action_space.shape[0]

        q_hat = self.dqn(encoded_state_action_space)

        assert q_hat.shape[0] == len(legal_moves)

        chosen_move_idx = torch.argmax(q_hat).item()

        return legal_moves[chosen_move_idx]

    def encode_board_state(self) -> torch.Tensor:
        # Encode the board state as a tensor
        return self.board.encode()

    def encode_state_action_(self) -> torch.Tensor:
        legal_moves = self.generate_legal_moves()

        encoded_legal_moves = self.board_state_encoder.buffalo_move_one_hot_encoder(legal_moves)
        encoded_board_state = self.encode_board_state()

        return torch.cat([encoded_board_state, encoded_legal_moves], dim=-1)
