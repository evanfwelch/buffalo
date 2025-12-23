import pytest

from buffalo.board import Board, Piece, PieceType, Player


def test_initial_setup():
    board = Board()

    assert board.current_player == Player.BUFFALO

    buffalo_positions = [
        pos for pos, piece in board.pieces.items() if piece.type == PieceType.BUFFALO
    ]
    assert len(buffalo_positions) == board.width
    assert all(y == 0 for _, y in buffalo_positions)

    dog_positions = [
        pos for pos, piece in board.pieces.items() if piece.type == PieceType.DOG
    ]
    assert sorted(dog_positions) == [(3, 5), (4, 5), (6, 5), (7, 5)]

    chief_positions = [
        pos for pos, piece in board.pieces.items() if piece.type == PieceType.CHIEF
    ]
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


def test_move_switches_player_and_hunter_moves_not_bottom():
    board = Board()

    moved = board.move_piece(0, 0, 0, 1)
    assert moved is True
    assert board.current_player == Player.HUNTERS

    for move in board.legal_moves():
        print(move)
        assert move.end.y != board.height - 1


def test_winner_detected_on_bottom_row():
    board = Board()
    board.pieces[(0, board.height - 1)] = Piece(PieceType.BUFFALO, Player.BUFFALO)

    assert board.check_for_winner() == Player.BUFFALO


def test_hunters_win_when_buffalo_no_moves():
    board = Board()
    board.pieces = {
        (0, 0): Piece(PieceType.BUFFALO, Player.BUFFALO),
        (0, 1): Piece(PieceType.DOG, Player.HUNTERS),
    }
    board.current_player = Player.BUFFALO

    assert board.legal_moves() == []
    assert board.check_for_winner() == Player.HUNTERS


@pytest.mark.parametrize(
    ("piece_type", "player", "start", "end", "expected"),
    [
        (PieceType.BUFFALO, Player.BUFFALO, (5, 0), (5, 1), True),
        (PieceType.BUFFALO, Player.BUFFALO, (5, 0), (5, 2), False),
        (PieceType.BUFFALO, Player.BUFFALO, (5, 0), (6, 1), False),
        (PieceType.CHIEF, Player.HUNTERS, (5, 5), (6, 6), True),
        (PieceType.CHIEF, Player.HUNTERS, (5, 5), (7, 6), False),
        (PieceType.CHIEF, Player.HUNTERS, (5, 5), (5, 6), False),
        (PieceType.DOG, Player.HUNTERS, (5, 5), (5, 4), True),
        (PieceType.DOG, Player.HUNTERS, (5, 5), (7, 5), True),
        (PieceType.DOG, Player.HUNTERS, (5, 5), (7, 6), True),
        (PieceType.DOG, Player.HUNTERS, (5, 5), (6, 4), False),
        (PieceType.DOG, Player.HUNTERS, (5, 5), (5, 0), False),
        (PieceType.DOG, Player.HUNTERS, (5, 5), (5, 5), False),
    ],
)
def test_single_piece_moves_on_empty_board(piece_type, player, start, end, expected):
    board = Board()
    board.pieces = {start: Piece(piece_type, player)}
    board.current_player = player

    moved = board.move_piece(start[0], start[1], end[0], end[1])
    assert moved is expected
