import logging
from typing import Optional, Tuple

import arcade
import click

from .board import (
    Board,
    MoveResult,
    PieceType,
    Player,
    Position,
    MoveRecord,
    MoveResult,
)
from .bots import NaiveBuffalo, NaiveHunter
from .game import Game

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOARD_WIDTH = 11
BOARD_HEIGHT = 7
SQUARE_SIZE = 80
PANEL_WIDTH = 360
PANEL_PADDING = 16
PANEL_BG = (30, 30, 30)
PANEL_TEXT = (230, 230, 230)
PANEL_HEADER = (255, 210, 120)
PANEL_LINE_HEIGHT = 18
PANEL_MAX_LINES = 28
BOARD_PIXEL_WIDTH = BOARD_WIDTH * SQUARE_SIZE
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
        show_logs: bool = False,
    ) -> None:
        window_width = BOARD_PIXEL_WIDTH + (PANEL_WIDTH if show_logs else 0)
        super().__init__(window_width, HEIGHT, "Buffalo!")
        self.selected_pos: Optional[Tuple[int, int]] = None
        self.buffalo_strategy = buffalo_strategy
        self.hunter_strategy = hunter_strategy
        self.show_logs = show_logs
        board = Board()
        self.game = Game(
            board=board,
            buffalo_controller=self._controller_for_strategy(board, Player.BUFFALO),
            hunter_controller=self._controller_for_strategy(board, Player.HUNTERS),
        )
        self.started = False
        self.frame = 0
        self.max_frames = max_frames
        self.bot_delay = 0.25
        self.bot_elapsed = 0.0
        self._history_cache = []
        self._history_count = 0

    def on_draw(self) -> None:
        self.clear()
        self.draw_board()
        if self.selected_pos:
            self.draw_selected(self.selected_pos)
        self.draw_pieces()
        if self.show_logs:
            self.draw_sidebar()

    def on_update(self, delta_time: float) -> None:
        if not self.started:
            return

        self.frame += 1
        if self.max_frames and self.frame >= self.max_frames:
            self.close()
            return

        controller = self.game.controller_for_current_player()

        assert controller is not None, "Controller should not be None"

        self.bot_elapsed += delta_time
        if self.bot_elapsed < self.bot_delay:
            return
        self.bot_elapsed = 0.0
        move_result, move_record = self.game.step()

        self._maybe_end_game(move_result)

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int) -> None:
        # start the game upon first click
        if not self.started:
            self.started = True
            return

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
        arcade.draw_line(
            0,
            HEIGHT - SQUARE_SIZE,
            BOARD_PIXEL_WIDTH,
            HEIGHT - SQUARE_SIZE,
            LINE_COLOR,
            4,
        )
        arcade.draw_line(0, SQUARE_SIZE, BOARD_PIXEL_WIDTH, SQUARE_SIZE, LINE_COLOR, 4)

    def draw_sidebar(self) -> None:
        arcade.draw_lbwh_rectangle_filled(
            BOARD_PIXEL_WIDTH,
            0,
            PANEL_WIDTH,
            HEIGHT,
            PANEL_BG,
        )

        header_x = BOARD_PIXEL_WIDTH + PANEL_PADDING
        header_y = HEIGHT - PANEL_PADDING
        arcade.draw_text(
            "Move History",
            header_x,
            header_y,
            PANEL_HEADER,
            font_size=16,
            anchor_x="left",
            anchor_y="top",
        )

        if len(self.game.history) != self._history_count:
            self._history_count = len(self.game.history)
            recent = self.game.history[-PANEL_MAX_LINES:]
            self._history_cache = [
                f"{record.move_number:03d} {record.player.name} {record.board_after}" for record in recent
            ]

        text_y = header_y - (PANEL_LINE_HEIGHT * 1.6)
        for line in self._history_cache:
            arcade.draw_text(
                line,
                header_x,
                text_y,
                PANEL_TEXT,
                font_size=12,
                anchor_x="left",
                anchor_y="top",
            )
            text_y -= PANEL_LINE_HEIGHT

    def draw_pieces(self) -> None:
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                piece = self.game.board.get_piece_at(x, y)
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

    def _controller_for_strategy(self, board: Board, player: Player):
        if player == Player.BUFFALO and self.buffalo_strategy == "naive":
            return NaiveBuffalo(board)
        if player == Player.HUNTERS and self.hunter_strategy == "naive":
            return NaiveHunter(board)
        return None

    def _maybe_end_game(self, move_result: MoveResult) -> None:
        if move_result.winner_after_move is not None:
            self.started = False
            self.set_caption(
                f"YEEHAW: {move_result.winner_after_move.name} wins! Reason = {move_result.game_over_reason}"
            )
        return None


@click.command()
@click.option(
    "--frames",
    "max_frames",
    type=int,
    default=None,
    help="Number of frames to run (for testing)",
)
@click.option(
    "--buffalo-strategy",
    type=str,
    default="naive",
    help="Strategy for the buffalo player (e.g., 'naive')",
)
@click.option(
    "--hunter-strategy",
    type=str,
    default="naive",
    help="Strategy for the hunter player (e.g., 'naive')",
)
@click.option("--show-logs", is_flag=True, help="Show move history sidebar.")
def main(
    max_frames=None,
    buffalo_strategy: str = None,
    hunter_strategy: str = None,
    show_logs: bool = False,
) -> None:
    GameWindow(
        max_frames=max_frames,
        buffalo_strategy=buffalo_strategy,
        hunter_strategy=hunter_strategy,
        show_logs=show_logs,
    )
    arcade.run()


if __name__ == "__main__":
    main()
