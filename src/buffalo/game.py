import argparse
import pygame
from .board import Board, PieceType, Player

BOARD_WIDTH = 11
BOARD_HEIGHT = 7
SQUARE_SIZE = 80
WIDTH = BOARD_WIDTH * SQUARE_SIZE
HEIGHT = BOARD_HEIGHT * SQUARE_SIZE
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
LINE_COLOR = (0, 0, 0)

PIECE_COLORS = {
    PieceType.BUFFALO: (139, 69, 19),  # Brown
    PieceType.DOG: (105, 105, 105),    # Gray
    PieceType.CHIEF: (255, 215, 0),    # Gold
}

TEXT_COLORS = {
    PieceType.BUFFALO: (255, 255, 255),  # White
    PieceType.DOG: (0, 0, 0),           # Black
    PieceType.CHIEF: (0, 0, 0),         # Black
}


def draw_board(screen):
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            rect = pygame.Rect(x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            color = LIGHT if (x + y) % 2 == 0 else DARK
            pygame.draw.rect(screen, color, rect)
    # bold lines separating first and seventh ranks
    pygame.draw.line(screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), 4)
    pygame.draw.line(screen, LINE_COLOR, (0, HEIGHT - SQUARE_SIZE), (WIDTH, HEIGHT - SQUARE_SIZE), 4)


def draw_pieces(screen, font, board):
    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            piece = board.get_piece_at(x, y)
            if piece:
                center = (x * SQUARE_SIZE + SQUARE_SIZE // 2, 
                         y * SQUARE_SIZE + SQUARE_SIZE // 2)
                pygame.draw.circle(screen, PIECE_COLORS[piece.type], center, SQUARE_SIZE // 3)
                text = font.render(piece.type.value, True, TEXT_COLORS[piece.type])
                screen.blit(text, text.get_rect(center=center))


def get_board_position(pos):
    x, y = pos
    return x // SQUARE_SIZE, y // SQUARE_SIZE


def draw_selected(screen, pos):
    x, y = pos
    rect = pygame.Rect(x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
    highlight = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    pygame.draw.rect(highlight, (255, 255, 0, 100), highlight.get_rect())
    screen.blit(highlight, rect)


def main(max_frames=None):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Buffalo!")
    font = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()
    
    board = Board()
    selected_pos = None
    
    running = True
    frame = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = get_board_position(event.pos)
                if selected_pos is None:
                    # Select piece if it belongs to current player
                    piece = board.get_piece_at(x, y)
                    if piece and piece.player == board.current_player:
                        selected_pos = (x, y)
                else:
                    # Try to move piece if destination is clicked
                    if (x, y) != selected_pos:
                        from_x, from_y = selected_pos
                        # TODO: Add move validation here when implemented in Board
                        board.move_piece(from_x, from_y, x, y)
                    selected_pos = None
        
        draw_board(screen)
        if selected_pos:
            draw_selected(screen, selected_pos)
        draw_pieces(screen, font, board)
        pygame.display.flip()
        clock.tick(30)
        frame += 1
        if max_frames and frame >= max_frames:
            running = False
    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Buffalo board demo")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to run (for testing)")
    args = parser.parse_args()
    main(max_frames=args.frames)
