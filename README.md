# Catan AI

A rules-complete Catan game engine built to train reinforcement learning agents and benchmark them against heuristic baselines.

## Quick Start

```bash
uv sync                  # install dependencies
uv run pytest            # run tests (121 tests, <1s)
uv run ruff check .      # lint
uv run ruff format .     # format
```

## Project Structure

```
src/catan/
  topology.py      # Hex grid: 54 vertices, 72 edges from 19 axial-coord hexes
  board.py         # Hex/Board dataclasses, standard layout, ports, token mapping
  resources.py     # Resource & Terrain enums, terrain-to-resource mapping
  player.py        # Player state: resources, buildings, dev cards, VP
  actions.py       # 15 action types as frozen dataclasses
  constants.py     # Building costs, piece limits, VP target
  game.py          # Game engine: phases, legal actions, apply, victory check
  dev_cards.py     # DevCardType enum, 25-card deck, largest army
  ports.py         # PortType enum, 9 ports (4 generic + 5 specialized)
  longest_road.py  # DFS longest-path computation
  game_runner.py   # run_game() and run_tournament() for batch simulation

src/catan/ai/
  agent.py         # Abstract Agent base class
  heuristic.py     # RandomAgent (uniform random from legal actions)

tests/
  helpers.py           # Shared fixtures: make_game, skip_to_main_phase, etc.
  test_topology.py     # Vertex/edge counts, adjacency, hex mappings
  test_board.py        # Board construction, hex layout
  test_placement.py    # Forward/backward order, distance rule, starting resources
  test_resources.py    # Production (settlement=1, city=2), robber blocking, desert
  test_building.py     # Settlement/city/road rules, costs, VP, connectivity
  test_robber.py       # Discard >7 cards, robber move, stealing
  test_dev_cards.py    # All 5 card types, timing rules, largest army
  test_trading.py      # 4:1/3:1/2:1 trade rates, port access
  test_longest_road.py # DFS paths, opponent blocking, VP awards
  test_victory.py      # Win at 10 VP, tie-breaking, max turns
  test_invariants.py   # State validator run after 20 full random games
  test_game.py         # Full game completion, legal action validity
```

## Game Rules Implemented

- **Placement phase** — 2 rounds (forward then reverse), distance rule, second round grants starting resources
- **Resource production** — settlements produce 1, cities produce 2, robber blocks hex
- **Building** — settlements, cities, roads with cost checks, distance rule, road connectivity
- **Robber** — roll 7 triggers discard (>7 cards, half rounded down) then move + steal
- **Development cards** — knight, VP, road building, year of plenty, monopoly; can't play card bought same turn
- **Largest army** — 2 VP for 3+ knights played
- **Longest road** — 2 VP for 5+ connected roads (DFS, opponent buildings block)
- **Ports** — 9 ports (4 generic 3:1, 5 specialized 2:1) affecting bank trade rates
- **Victory** — first to 10 VP wins; 300 turn limit

## Roadmap

- [x] **Phase 1** — Rules-complete game engine with 121 tests
- [ ] **Phase 2** — Gymnasium env, observation encoding, action space, Maskable PPO
- [ ] **Phase 3** — Training curriculum, heuristic baselines, evaluation & visualization
- [ ] **Phase 4** — Polish, results charts, README showcase
