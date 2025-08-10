import unittest
from buffalo.board import Board, Player, Position

class TestBoard(unittest.TestCase):
    def test_invalid_move_hunters_piece_on_buffalos_turn(self):
        # Initialize the board
        board = Board()
        
        # Ensure it's buffalo's turn
        self.assertEqual(board.current_player, Player.BUFFALO)
        
        # Get the initial state of the board
        initial_pieces = board.pieces.copy()
        
        # Attempt to move a hunter's piece (e.g., dog at (0, 6))
        from_pos = Position(1, 6)
        to_pos = Position(1, 5)
        move_result = board.move_piece(from_pos.x, from_pos.y, to_pos.x, to_pos.y)
        
        # Assert the move is invalid
        self.assertFalse(move_result)
        
        # Assert the board state has not changed
        self.assertEqual(board.pieces, initial_pieces)

if __name__ == "__main__":
    unittest.main()
