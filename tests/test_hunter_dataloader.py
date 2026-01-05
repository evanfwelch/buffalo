import torch

from buffalo.board import Board, MoveRecord, PieceType, Player, Position
from buffalo.dataloader import HunterGameDataset


def test_hunter_dataloader_encodes_action(tmp_path):
    board = Board()
    board.move_piece(0, 0, 0, 1)
    board_after_buffalo = board.serialize()
    board.move_piece(5, 5, 5, 4)
    board_after_hunter = board.serialize()
    board.move_piece(1, 0, 1, 1)
    board_after_buffalo_two = board.serialize()

    def snapshot_pieces(board: Board):
        return {(x, y): piece for (x, y), piece in board.pieces.items()}

    lines = [
        MoveRecord(
            move_number=1,
            player=Player.BUFFALO,
            piece_type=PieceType.BUFFALO,
            from_pos=Position(0, 0),
            to_pos=Position(0, 1),
            pieces_before=snapshot_pieces(Board()),
            pieces_after=snapshot_pieces(Board.deserialize(board_after_buffalo)),
            captured_piece=None,
            winner_after_move=None,
            game_over_reason=None,
        ).to_json(),
        MoveRecord(
            move_number=2,
            player=Player.HUNTERS,
            piece_type=PieceType.CHIEF,
            from_pos=Position(5, 5),
            to_pos=Position(5, 4),
            pieces_before=snapshot_pieces(Board.deserialize(board_after_buffalo)),
            pieces_after=snapshot_pieces(Board.deserialize(board_after_hunter)),
            captured_piece=None,
            winner_after_move=None,
            game_over_reason=None,
        ).to_json(),
        MoveRecord(
            move_number=3,
            player=Player.BUFFALO,
            piece_type=PieceType.BUFFALO,
            from_pos=Position(1, 0),
            to_pos=Position(1, 1),
            pieces_before=snapshot_pieces(Board.deserialize(board_after_hunter)),
            pieces_after=snapshot_pieces(Board.deserialize(board_after_buffalo_two)),
            captured_piece=None,
            winner_after_move=None,
            game_over_reason=None,
        ).to_json(),
    ]

    jsonl_path = tmp_path / "game-1.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")

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
