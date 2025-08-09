import argparse
import pygame

BOARD_WIDTH = 11
BOARD_HEIGHT = 7
SQUARE_SIZE = 80
WIDTH = BOARD_WIDTH * SQUARE_SIZE
HEIGHT = BOARD_HEIGHT * SQUARE_SIZE
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
LINE_COLOR = (0, 0, 0)


def draw_board(screen):
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            rect = pygame.Rect(x * SQUARE_SIZE, y * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            color = LIGHT if (x + y) % 2 == 0 else DARK
            pygame.draw.rect(screen, color, rect)
    # bold lines separating first and seventh ranks
    pygame.draw.line(screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), 4)
    pygame.draw.line(screen, LINE_COLOR, (0, HEIGHT - SQUARE_SIZE), (WIDTH, HEIGHT - SQUARE_SIZE), 4)


def draw_pieces(screen, font):
    # buffalo pawns on top rank
    for x in range(BOARD_WIDTH):
        center = (x * SQUARE_SIZE + SQUARE_SIZE // 2, SQUARE_SIZE // 2)
        pygame.draw.circle(screen, (139, 69, 19), center, SQUARE_SIZE // 3)
        text = font.render("B", True, (255, 255, 255))
        screen.blit(text, text.get_rect(center=center))

    # dogs on bottom rank
    dog_positions = [0, 3, 7, 10]
    for x in dog_positions:
        center = (x * SQUARE_SIZE + SQUARE_SIZE // 2, HEIGHT - SQUARE_SIZE // 2)
        pygame.draw.circle(screen, (105, 105, 105), center, SQUARE_SIZE // 3)
        text = font.render("D", True, (0, 0, 0))
        screen.blit(text, text.get_rect(center=center))

    # chief at center bottom
    x = BOARD_WIDTH // 2
    center = (x * SQUARE_SIZE + SQUARE_SIZE // 2, HEIGHT - SQUARE_SIZE // 2)
    pygame.draw.circle(screen, (255, 215, 0), center, SQUARE_SIZE // 3)
    text = font.render("C", True, (0, 0, 0))
    screen.blit(text, text.get_rect(center=center))


def main(max_frames=None):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Buffalo")
    font = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()
    running = True
    frame = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        draw_board(screen)
        draw_pieces(screen, font)
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
