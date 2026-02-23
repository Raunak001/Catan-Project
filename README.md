# Catan AI

A rules-complete Catan game engine built to train reinforcement learning agents and benchmark them against heuristic baselines.

## Quick Start

```bash
uv sync                  # install dependencies
uv run pytest            # run tests (149 tests, ~1s)
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
  gym_env.py       # Gymnasium env: obs encoding (407 floats), action space (463 discrete)
  train.py         # Maskable PPO training script (sb3-contrib)

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
  test_gym_env.py      # Gymnasium env: obs/action encoding, full game rollouts
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

## RL Environment

Single-agent Gymnasium environment (`CatanEnv`). The training agent sits at seat 0; seats 1–3 are controlled by opponent agents (default: `RandomAgent`). Future work will copy the trained model into opponent seats for self-play (4 AIs against each other).

- **Observation**: 407 floats — hex terrain/tokens, vertex/edge ownership, player resources/VP/dev cards, ports, phase, turn
- **Action space**: 463 discrete — settlements (54), roads (72), cities (54), bank trades (20), dev cards, robber, discard (50 dynamic), end turn
- **Action masking**: boolean mask from `legal_actions()`, compatible with sb3-contrib `MaskablePPO`
- **Reward**: +1 win, -1 loss, 0 otherwise

```bash
uv run python -m catan.ai.train --timesteps 100000   # train PPO agent
```

## Roadmap

- [x] **Phase 1** — Rules-complete game engine with 121 tests
- [x] **Phase 2** — Gymnasium env (407-dim obs, 463 actions), Maskable PPO scaffold, 28 env tests
- [ ] **Phase 3** — Training curriculum, heuristic baselines, self-play, evaluation & visualization
- [ ] **Phase 4** — Polish, results charts, README showcase
