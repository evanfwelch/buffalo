import unittest

from buffalo.board import Board, Position
from buffalo.bots import NaiveBuffalo


class TestBot(unittest.TestCase):
    def test_generate_legal_moves_initial_buffalo(self):
        board = Board()
        bot = NaiveBuffalo(board)
        moves = bot.generate_legal_moves()

        self.assertEqual(len(moves), board.width)

        expected = {
            (x, 0, x, 1)
            for x in range(board.width)
        }
        got = {
            (from_pos.x, from_pos.y, to_pos.x, to_pos.y)
            for from_pos, to_pos in moves
        }
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
