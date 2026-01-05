"""Random game simulation and CSV export for Buffalo."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional
import importlib

import click

from .board import Board
from .game import Game, MoveRecord


@click.command()
@click.option("--num-games", type=int, default=10, show_default=True)
@click.option("--output-dir", type=str, default="simulated_games", show_default=True)
@click.option("--simulation-name", type=str, default="test1", show_default=True)
@click.option("--max-moves", type=int, default=5000, show_default=True)
@click.option("--seed", type=int, default=None)
@click.option("--buffalo-strategy", type=str, default="NaiveBuffalo", show_default=True)
@click.option("--hunter-strategy", type=str, default="NaiveHunter", show_default=True)
def main(
    num_games: int,
    output_dir: str,
    simulation_name: str,
    max_moves: int,
    seed: Optional[int],
    buffalo_strategy: str,
    hunter_strategy: str,
) -> None:
    seed = random.Random(seed)

    bots_module = importlib.import_module("buffalo.bots")
    buffalo_strategy_clazz = getattr(bots_module, buffalo_strategy)
    hunter_strategy_clazz = getattr(bots_module, hunter_strategy)

    for i_game in range(num_games):
        board = Board()

        buffalo_controller = buffalo_strategy_clazz(board)
        hunter_controller = hunter_strategy_clazz(board)
        assert buffalo_controller.board == hunter_controller.board
        assert buffalo_controller.board == board

        game = Game(buffalo_controller=buffalo_controller, hunter_controller=hunter_controller, board=board)

        while not game.game_over and game.board.move_number < max_moves:
            game.step()

        if game.board.move_number == max_moves:
            print(f"Game {i_game} reached max moves of {max_moves} without a winner, skipping for now.")

        # TODO: log the game in the desired subfolder
        game_filename: str = f"game-{i_game:09d}.jsonl"
        game_folder: str = os.path.join(output_dir, simulation_name)
        os.makedirs(game_folder, exist_ok=True)
        game_path: str = os.path.join(game_folder, game_filename)

        game.write_history(game_path)


if __name__ == "__main__":
    main()
