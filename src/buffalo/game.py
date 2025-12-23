import argparse
import logging
from typing import Optional, Tuple

import arcade

from .board import Board, PieceType, Player
from .bots import NaiveBuffalo, NaiveHunter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOARD_WIDTH = 11
BOARD_HEIGHT = 7
SQUARE_SIZE = 80
WIDTH = BOARD_WIDTH * SQUARE_SIZE
HEIGHT = BOARD_HEIGHT * SQUARE_SIZE


LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
LINE_COLOR = (0, 0, 0)

PIECE_COLORS = {
    PieceType.BUFFALO: (139, 69, 19),  # Brown
    PieceType.DOG: (105, 105, 105),  # Gray
    PieceType.CHIEF: (255, 215, 0),  # Gold
}

TEXT_COLORS = {
    PieceType.BUFFALO: (255, 255, 255),  # White
    PieceType.DOG: (0, 0, 0),  # Black
    PieceType.CHIEF: (0, 0, 0),  # Black
}


def to_screen_center(x: int, y: int) -> Tuple[float, float]:
    return (
        x * SQUARE_SIZE + SQUARE_SIZE / 2,
        HEIGHT - (y * SQUARE_SIZE + SQUARE_SIZE / 2),
    )


def to_board_position(screen_x: float, screen_y: float) -> Tuple[int, int]:
    return (
        int(screen_x // SQUARE_SIZE),
        int((HEIGHT - screen_y) // SQUARE_SIZE),
    )


class GameWindow(arcade.Window):
    def __init__(
        self,
        max_frames: Optional[int] = None,
        buffalo_strategy: Optional[str] = None,
        hunter_strategy: Optional[str] = None,
    ) -> None:
        super().__init__(WIDTH, HEIGHT, "Buffalo!")
        self.board = Board()
        self.selected_pos: Optional[Tuple[int, int]] = None
        self.buffalo_strategy = buffalo_strategy
        self.hunter_strategy = hunter_strategy
        self.buffalo_bot = NaiveBuffalo(self.board) if buffalo_strategy == "naive" else None
        self.hunter_bot = NaiveHunter(self.board) if hunter_strategy == "naive" else None
        self.started = False
        self.frame = 0
        self.max_frames = max_frames
        self.bot_delay = 0.25
        self.bot_elapsed = 0.0

    def on_draw(self) -> None:
        self.clear()
        self.draw_board()
        if self.selected_pos:
            self.draw_selected(self.selected_pos)
        self.draw_pieces()

    def on_update(self, delta_time: float) -> None:
        if not self.started:
            return

        self.frame += 1
        if self.max_frames and self.frame >= self.max_frames:
            self.close()
            return

        current_player = self.board.current_player
        bot = self._bot_for_player(current_player)
        if bot is not None:
            self.bot_elapsed += delta_time
            if self.bot_elapsed < self.bot_delay:
                return
            self.bot_elapsed = 0.0
            made_move = bot.make_move()
            self._check_game_end(made_move, used_bot=True)
        else:
            self._check_game_end(None, used_bot=False)

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int) -> None:
        if not self.started:
            self.started = True
            return

        if self._bot_for_player(self.board.current_player) is not None:
            return

        board_x, board_y = to_board_position(x, y)
        if not (0 <= board_x < BOARD_WIDTH and 0 <= board_y < BOARD_HEIGHT):
            return

        if self.selected_pos is None:
            piece = self.board.get_piece_at(board_x, board_y)
            if piece and piece.player == self.board.current_player:
                self.selected_pos = (board_x, board_y)
            return

        if (board_x, board_y) != self.selected_pos:
            from_x, from_y = self.selected_pos
            self.board.move_piece(from_x, from_y, board_x, board_y)
        self.selected_pos = None

    def draw_board(self) -> None:
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                color = LIGHT if (x + y) % 2 == 0 else DARK
                center_x, center_y = to_screen_center(x, y)
                arcade.draw_lbwh_rectangle_filled(
                    center_x - SQUARE_SIZE / 2,
                    center_y - SQUARE_SIZE / 2,
                    SQUARE_SIZE,
                    SQUARE_SIZE,
                    color,
                )
        arcade.draw_line(0, HEIGHT - SQUARE_SIZE, WIDTH, HEIGHT - SQUARE_SIZE, LINE_COLOR, 4)
        arcade.draw_line(0, SQUARE_SIZE, WIDTH, SQUARE_SIZE, LINE_COLOR, 4)

    def draw_pieces(self) -> None:
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                piece = self.board.get_piece_at(x, y)
                if piece:
                    center_x, center_y = to_screen_center(x, y)
                    arcade.draw_circle_filled(
                        center_x,
                        center_y,
                        SQUARE_SIZE / 3,
                        PIECE_COLORS[piece.type],
                    )
                    arcade.draw_text(
                        piece.type.value,
                        center_x,
                        center_y,
                        TEXT_COLORS[piece.type],
                        font_size=18,
                        anchor_x="center",
                        anchor_y="center",
                    )

    def draw_selected(self, pos: Tuple[int, int]) -> None:
        x, y = pos
        center_x, center_y = to_screen_center(x, y)
        arcade.draw_lbwh_rectangle_filled(
            center_x - SQUARE_SIZE / 2,
            center_y - SQUARE_SIZE / 2,
            SQUARE_SIZE,
            SQUARE_SIZE,
            (255, 255, 0, 100),
        )

    def _bot_for_player(self, player: Player):
        if player == Player.BUFFALO:
            return self.buffalo_bot
        return self.hunter_bot

    def _check_game_end(self, made_move: Optional[bool], used_bot: bool) -> None:
        maybe_winner = self.board.check_for_winner()
        if maybe_winner is not None:
            self.started = False
            self.set_caption(f"YEEHAW: {maybe_winner.name} wins!")
            return
        if used_bot and made_move is False:
            self.started = False
            self.set_caption("No valid moves -- Game Over!")


def main(max_frames=None, buffalo_strategy: str = None, hunter_strategy: str = None) -> None:
    GameWindow(
        max_frames=max_frames,
        buffalo_strategy=buffalo_strategy,
        hunter_strategy=hunter_strategy,
    )
    arcade.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buffalo board demo")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to run (for testing)")
    parser.add_argument(
        "--buffalo-strategy",
        type=str,
        default="naive",
        help="Strategy for the buffalo player (e.g., 'naive')",
    )
    parser.add_argument(
        "--hunter-strategy",
        type=str,
        default="naive",
        help="Strategy for the hunter player (e.g., 'naive')",
    )
    args = parser.parse_args()
    main(
        max_frames=args.frames,
        buffalo_strategy=args.buffalo_strategy,
        hunter_strategy=args.hunter_strategy,
    )
