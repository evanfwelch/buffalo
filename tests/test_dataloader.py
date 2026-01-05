import torch

from buffalo.board import Board, MoveRecord, PieceType, Player, Position
from buffalo.dataloader import BuffaloGameDataset
from buffalo.encoders import BoardStateEncoder


def test_dataloader_reads_buffalo_hunter_pairs(tmp_path):
    board = Board()
    board_before = board.serialize()
    board.move_piece(0, 0, 0, 1)
    board_after_buffalo = board.serialize()
    board.move_piece(3, 5, 3, 4)
    board_after_hunter = board.serialize()

    def snapshot_pieces(board: Board):
        return {(x, y): piece for (x, y), piece in board.pieces.items()}

    lines = [
        MoveRecord(
            move_number=1,
            player=Player.BUFFALO,
            piece_type=PieceType.BUFFALO,
            from_pos=Position(0, 0),
            to_pos=Position(0, 1),
            pieces_before=snapshot_pieces(Board.deserialize(board_before)),
            pieces_after=snapshot_pieces(Board.deserialize(board_after_buffalo)),
            captured_piece=None,
            winner_after_move=None,
            game_over_reason=None,
        ).to_json(),
        MoveRecord(
            move_number=2,
            player=Player.HUNTERS,
            piece_type=PieceType.DOG,
            from_pos=Position(3, 5),
            to_pos=Position(3, 4),
            pieces_before=snapshot_pieces(Board.deserialize(board_after_buffalo)),
            pieces_after=snapshot_pieces(Board.deserialize(board_after_hunter)),
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
