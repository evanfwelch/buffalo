import pytest

from buffalo.board import Board, Piece, PieceType, Player, GameOverReason


def test_initial_setup():
    board = Board()

    assert board.current_player == Player.BUFFALO

    buffalo_positions = [pos for pos, piece in board.pieces.items() if piece.type == PieceType.BUFFALO]
    assert len(buffalo_positions) == board.width
    assert all(y == 0 for _, y in buffalo_positions)

    dog_positions = [pos for pos, piece in board.pieces.items() if piece.type == PieceType.DOG]
    assert sorted(dog_positions) == [(3, 5), (4, 5), (6, 5), (7, 5)]

    chief_positions = [pos for pos, piece in board.pieces.items() if piece.type == PieceType.CHIEF]
    assert chief_positions == [(5, 5)]


def test_buffalo_legal_moves_initial():
    board = Board()

    legal_moves = board.legal_moves()
    assert len(legal_moves) == board.width

    for move in legal_moves:
        assert move.start.y == 0
        assert move.end.y == 1
        piece = board.get_piece_at(move.start.x, move.start.y)
        assert piece is not None
        assert piece.player == Player.BUFFALO


def test_winner_detected_on_bottom_row():
    board = Board()
    board.pieces[(0, board.height - 1)] = Piece(PieceType.BUFFALO, Player.BUFFALO)

    assert board.check_for_winner() == (Player.BUFFALO, GameOverReason.BUFFALO_CROSSED)


def test_hunters_win_when_buffalo_no_moves():
    board = Board()
    board.pieces = {
        (0, 0): Piece(PieceType.BUFFALO, Player.BUFFALO),
        (0, 1): Piece(PieceType.DOG, Player.HUNTERS),
    }
    board.current_player = Player.BUFFALO

    assert board.legal_moves() == []
    assert board.check_for_winner() == (Player.HUNTERS, GameOverReason.BUFFALO_STUCK)


@pytest.mark.parametrize(
    ("piece_type", "player", "start", "end", "expected"),
    [
        (
            PieceType.BUFFALO,
            Player.BUFFALO,
            (5, 0),
            (5, 1),
            True,
        ),  # can move down by 1step
        (
            PieceType.BUFFALO,
            Player.BUFFALO,
            (5, 0),
            (5, 2),
            False,
        ),  # cant move by 2 steps
        (
            PieceType.BUFFALO,
            Player.BUFFALO,
            (5, 0),
            (6, 1),
            False,
        ),  # cant move diagonally
        (
            PieceType.BUFFALO,
            Player.BUFFALO,
            (5, 1),
            (5, 0),
            False,
        ),  # buffalo cant move backward
        (
            PieceType.BUFFALO,
            Player.BUFFALO,
            (5, 5),
            (5, 6),
            True,
        ),  # only buffalo can cross to bottom rank for a win
        (
            PieceType.CHIEF,
            Player.HUNTERS,
            (5, 5),
            (6, 4),
            True,
        ),  # can move diagonally by 1
        (
            PieceType.CHIEF,
            Player.HUNTERS,
            (5, 5),
            (7, 6),
            False,
        ),  # cant move by 2 spaces
        (
            PieceType.CHIEF,
            Player.HUNTERS,
            (5, 5),
            (5, 6),
            False,
        ),  # cannot move to bottom rank
        (
            PieceType.CHIEF,
            Player.HUNTERS,
            (5, 1),
            (5, 0),
            False,
        ),  # cannot move to top rank
        (PieceType.DOG, Player.HUNTERS, (5, 5), (5, 4), True),
        (PieceType.DOG, Player.HUNTERS, (5, 5), (7, 5), True),
        (
            PieceType.DOG,
            Player.HUNTERS,
            (5, 5),
            (7, 6),
            False,
        ),  # cannot move to bottom rank
        (PieceType.DOG, Player.HUNTERS, (5, 5), (6, 4), True),
        (PieceType.DOG, Player.HUNTERS, (5, 5), (5, 5), False),
    ],
)
def test_single_piece_moves_on_empty_board(piece_type, player, start, end, expected):
    board = Board()
    board.pieces = {start: Piece(piece_type, player)}
    board.current_player = player

    is_valid = board._is_valid_move(board.get_piece_at(*start), start[0], start[1], end[0], end[1])
    assert is_valid is expected


def test_serialize_roundtrip_initial_board():
    board = Board()

    serialized = board.serialize()
    restored = Board.deserialize(serialized)

    assert restored.serialize() == serialized
    assert restored.current_player == Player.BUFFALO


def test_serialize_roundtrip_custom_board():
    board = Board()
    board.pieces = {
        (0, 0): Piece(PieceType.BUFFALO, Player.BUFFALO),
        (5, 3): Piece(PieceType.DOG, Player.HUNTERS),
        (10, 6): Piece(PieceType.CHIEF, Player.HUNTERS),
    }

    serialized = board.serialize()
    restored = Board.deserialize(serialized)

    assert restored.serialize() == serialized
    assert restored.pieces == board.pieces


def test_deserialize_rejects_invalid_data():
    with pytest.raises(ValueError):
        Board.deserialize("too/few/rows")

    board = Board()
    bad_row = "." * (board.width - 1)
    rows = [bad_row for _ in range(board.height)]
    with pytest.raises(ValueError):
        Board.deserialize("/".join(rows))

    rows = ["." * board.width for _ in range(board.height)]
    rows[0] = "X" + "." * (board.width - 1)
    with pytest.raises(ValueError):
        Board.deserialize("/".join(rows))
