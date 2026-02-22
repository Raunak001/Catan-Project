# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                          # Install/update dependencies
uv run pytest                    # Run all tests
uv run pytest tests/test_board.py          # Run a single test file
uv run pytest tests/test_board.py::test_standard_board_has_19_hexes  # Run a single test
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run ruff check --fix .        # Lint with auto-fix
```

## Architecture

This is a Catan AI/simulator built in Python using a `src` layout.

**Core engine** (`src/catan/`):
- `resources.py` — `Resource` and `Terrain` enums, terrain-to-resource mapping
- `board.py` — `Hex` and `Board` dataclasses; `Board.standard()` creates the 19-hex layout using axial coordinates
- `player.py` — `Player` state (resources, victory points)
- `game.py` — `Game` orchestrates board, players, turns, and dice rolls

**AI layer** (`src/catan/ai/`):
- `agent.py` — Abstract `Agent` base class; implement `choose_action(game)` and `name()` to create new AI strategies

**Data flow**: `Game` holds a `Board` (list of `Hex`) and `Player` list. AI agents receive the full `Game` state and mutate it through actions.

## Permissions

```json
{
  "permissions": {
    "allow": [
      "Bash(uv *)",
      "Bash(git *)",
      "Bash(python *)"
    ]
  }
}
```

## Conventions

- Python 3.12+, type hints everywhere
- Dataclasses for state, enums for fixed categories
- Ruff for linting (`E`, `F`, `I`, `UP` rules) and formatting, line length 99
- Tests live in `tests/`, mirroring `src/catan/` structure
