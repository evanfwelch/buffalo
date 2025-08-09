# buffalo
Recreating and hosting the Buffalo boardgame from our 2025 Ski Trip

## Local Prototype

This repo contains a simple Pygame prototype of the Buffalo board. The board is 11 squares wide by 7 tall. The top rank holds 11 buffalo pawns. The bottom rank holds the opposing chief and four dogs.

### Setup

This project uses [Poetry](https://python-poetry.org/).

```bash
pip install --user poetry  # or use pipx
poetry install
```

### Run

```bash
poetry run python -m buffalo.game
```

This opens a window showing the board and starting pieces.

### Test

To verify the script without opening a window (useful for CI or local smoke tests), render a single frame using a headless SDL driver:

```bash
SDL_VIDEODRIVER=dummy poetry run python -m buffalo.game --frames 1
```
