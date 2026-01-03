import torch

from buffalo.board import Board, Move, Piece, PieceType, Player, Position
from buffalo.encoders import BoardStateEncoder


def test_encode_marks_piece_positions_and_turn():
    board = Board()
    board.pieces = {
        (0, 0): Piece(PieceType.BUFFALO, Player.BUFFALO),
        (5, 3): Piece(PieceType.DOG, Player.HUNTERS),
        (10, 6): Piece(PieceType.CHIEF, Player.HUNTERS),
    }
    board.current_player = Player.HUNTERS

    encoder = BoardStateEncoder(board_width=board.width, board_height=board.height)
    state = encoder.encode(board)

    assert state.shape == (encoder.state_size,)

    def idx(x: int, y: int, piece_type: PieceType) -> int:
        square_index = y * board.width + x
        type_index = encoder.piece_types.index(piece_type)
        return square_index * len(encoder.piece_types) + type_index

    assert state[idx(0, 0, PieceType.BUFFALO)] == 1.0
    assert state[idx(5, 3, PieceType.DOG)] == 1.0
    assert state[idx(10, 6, PieceType.CHIEF)] == 1.0
    assert state[-1] == 1.0
    assert torch.sum(state).item() == 4.0


def test_encode_empty_board_tracks_turn_only():
    board = Board()
    board.pieces = {}
    board.current_player = Player.BUFFALO

    encoder = BoardStateEncoder(board_width=board.width, board_height=board.height)
    state = encoder.encode(board)

    assert state[-1] == 0.0
    assert torch.sum(state).item() == 0.0


def test_buffalo_move_one_hot_encoder_files():
    encoder = BoardStateEncoder()
    moves = [
        Move(
            player=Player.BUFFALO,
            piece=Piece(PieceType.BUFFALO, Player.BUFFALO),
            start=Position(0, 0),
            end=Position(0, 1),
        ),
        Move(
            player=Player.BUFFALO,
            piece=Piece(PieceType.BUFFALO, Player.BUFFALO),
            start=Position(5, 2),
            end=Position(5, 3),
        ),
    ]

    one_hot = encoder.buffalo_move_one_hot_encoder(moves)

    assert one_hot.shape == (len(moves), encoder.board_width)
    assert one_hot[0, 0].item() == 1.0
    assert one_hot[1, 5].item() == 1.0
    assert torch.sum(one_hot).item() == 2.0


def test_buffalo_move_one_hot_encoder_empty():
    # NOTE: we can probably delete this as we should never be invoking with no legal moves
    encoder = BoardStateEncoder()

    one_hot = encoder.buffalo_move_one_hot_encoder([])

    assert one_hot.shape == (0, encoder.board_width)
    assert torch.sum(one_hot).item() == 0.0


def test_buffalo_joint_state_action_encoder_broadcasts_state():
    encoder = BoardStateEncoder()
    encoded_state = torch.zeros(encoder.state_size, dtype=torch.float32)
    encoded_state[0] = 1.0
    encoded_state[-1] = 1.0
    encoded_actions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    joint = encoder.buffalo_joint_state_action_encoder(
        encoded_state,
        encoded_actions,
    )

    expanded_state = encoded_state.unsqueeze(0).expand(encoded_actions.size(0), -1)
    expected = torch.cat([expanded_state, encoded_actions], dim=1)
    assert torch.equal(joint, expected)
