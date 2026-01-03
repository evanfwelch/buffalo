import csv

import torch

from buffalo.board import Board, MoveRecord
from buffalo.dataloader import HunterGameDataset


def test_hunter_dataloader_encodes_action(tmp_path):
    board = Board()
    board.move_piece(0, 0, 0, 1)
    board_after_buffalo = board.serialize()
    board.move_piece(5, 5, 5, 4)
    board_after_hunter = board.serialize()
    board.move_piece(1, 0, 1, 1)
    board_after_buffalo_two = board.serialize()

    rows = [
        {
            "move_number": "1",
            "player": "Player.BUFFALO",
            "piece_type": "PieceType.BUFFALO",
            "from_x": "0",
            "from_y": "0",
            "to_x": "0",
            "to_y": "1",
            "captured_piece": "",
            "winner_after_move": "",
            "game_over_reason": "",
            "board_before": Board().serialize(),
            "board_after": board_after_buffalo,
        },
        {
            "move_number": "2",
            "player": "Player.HUNTERS",
            "piece_type": "PieceType.CHIEF",
            "from_x": "5",
            "from_y": "5",
            "to_x": "5",
            "to_y": "4",
            "captured_piece": "",
            "winner_after_move": "",
            "game_over_reason": "",
            "board_before": board_after_buffalo,
            "board_after": board_after_hunter,
        },
        {
            "move_number": "3",
            "player": "Player.BUFFALO",
            "piece_type": "PieceType.BUFFALO",
            "from_x": "1",
            "from_y": "0",
            "to_x": "1",
            "to_y": "1",
            "captured_piece": "",
            "winner_after_move": "",
            "game_over_reason": "",
            "board_before": board_after_hunter,
            "board_after": board_after_buffalo_two,
        },
    ]

    csv_path = tmp_path / "game-1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MoveRecord.csv_fields())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    dataset = HunterGameDataset(tmp_path)
    samples = list(dataset)

    assert len(samples) == 1
    _, action, reward, _ = samples[0]

    assert action.shape == (8 + 77 + 77,)
    assert action[0].item() == 1.0
    assert torch.sum(action[:8]).item() == 1.0

    dogs_before = action[8 : 8 + 77]
    dogs_after = action[8 + 77 :]
    assert torch.sum(dogs_before).item() == 4.0
    assert torch.sum(dogs_after).item() == 4.0
    assert reward == 0.0
