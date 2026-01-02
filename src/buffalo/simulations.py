"""Random game simulation and CSV export for Buffalo."""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from typing import Optional

import click

from .game import Game, MoveRecord, csv_fields, record_to_row


@dataclass(frozen=True)
class SimulationResult:
    games: int
    output_dir: str


class RandomMoveController:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def choose_move(self, game: Game):
        legal_moves = game.legal_moves()
        if not legal_moves:
            return None
        return self.rng.choice(legal_moves)


class RandomGameSimulator:
    """Generate random games and export move logs to CSV files."""

    def __init__(self, seed: Optional[random.Random] = None) -> None:
        self.seed = seed or random.Random()

    def generate_games(
        self,
        num_games: int,
        output_dir: str = "random_games",
        simulation_name: str = "random",
        max_moves: int = 500,
    ) -> SimulationResult:
        os.makedirs(output_dir, exist_ok=True)
        for game_number in range(1, num_games + 1):
            filename = f"{simulation_name}-game-{game_number}.csv"
            path = os.path.join(output_dir, filename)
            self._write_game(path, max_moves)
        return SimulationResult(games=num_games, output_dir=output_dir)

    def _write_game(self, path: str, max_moves: int) -> None:
        game = Game(
            buffalo_controller=RandomMoveController(self.rng),
            hunter_controller=RandomMoveController(self.rng),
        )
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=csv_fields())
            writer.writeheader()
            self._play_game(game, writer, max_moves)

    def _play_game(self, game: Game, writer: csv.DictWriter, max_moves: int) -> None:
        while not game.game_over and game.move_number < max_moves:
            record = game.step()
            if record is None:
                return
            writer.writerow(record_to_row(record))

        if game.game_over:
            return

        board_state = game.board.serialize()
        record = MoveRecord(
            move_number=game.move_number,
            player=game.board.current_player,
            piece_type=None,
            from_pos=None,
            to_pos=None,
            board_before=board_state,
            board_after=board_state,
            captured_piece=None,
            legal_moves=0,
            move_made=False,
            game_over=True,
            winner=None,
            game_over_reason="max_moves",
        )
        writer.writerow(record_to_row(record))


@click.command()
@click.option("--games", "num_games", type=int, default=10, show_default=True)
@click.option("--output-dir", type=str, default="random_games", show_default=True)
@click.option("--simulation-name", type=str, default="random", show_default=True)
@click.option("--max-moves", type=int, default=500, show_default=True)
@click.option("--seed", type=int, default=None)
def main(
    num_games: int,
    output_dir: str,
    simulation_name: str,
    max_moves: int,
    seed: Optional[int],
) -> None:
    seed = random.Random(seed)
    simulator = RandomGameSimulator(seed=seed)
    simulator.generate_games(
        num_games=num_games,
        output_dir=output_dir,
        simulation_name=simulation_name,
        max_moves=max_moves,
    )


if __name__ == "__main__":
    main()
