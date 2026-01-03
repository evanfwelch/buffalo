import csv

import torch

from buffalo.board import Board, MoveRecord
from buffalo.dataloader import BuffaloGameDataset
from buffalo.encoders import BoardStateEncoder


def test_dataloader_reads_buffalo_hunter_pairs(tmp_path):
    board = Board()
    board_before = board.serialize()
    board.move_piece(0, 0, 0, 1)
    board_after_buffalo = board.serialize()
    board.move_piece(3, 5, 3, 4)
    board_after_hunter = board.serialize()

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
            "board_before": board_before,
            "board_after": board_after_buffalo,
        },
        {
            "move_number": "2",
            "player": "Player.HUNTERS",
            "piece_type": "PieceType.DOG",
            "from_x": "3",
            "from_y": "5",
            "to_x": "3",
            "to_y": "4",
            "captured_piece": "",
            "winner_after_move": "",
            "game_over_reason": "",
            "board_before": board_after_buffalo,
            "board_after": board_after_hunter,
        },
    ]

    csv_path = tmp_path / "game-1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MoveRecord.csv_fields())
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    dataset = BuffaloGameDataset(tmp_path)
    samples = list(dataset)

    assert len(samples) == 1
    state, action, reward, next_state = samples[0]

    encoder = BoardStateEncoder()
    expected_state = encoder.encode(Board.deserialize(board_before))
    expected_next_state = encoder.encode(Board.deserialize(board_after_hunter))

    assert torch.equal(state, expected_state)
    assert torch.equal(next_state, expected_next_state)
    assert action.shape == (encoder.board_width,)
    assert action[0].item() == 1.0
    assert torch.sum(action).item() == 1.0
    assert reward == 0.0
