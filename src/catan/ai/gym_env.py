"""Gymnasium environment wrapping the Catan game engine.

Single-agent environment: the training agent sits at seat 0, three opponent
agents (default RandomAgent) auto-play their turns. Future work will swap
opponents for copies of the trained model to enable self-play.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from catan.actions import (
    Action,
    BankTrade,
    BuildCity,
    BuildRoad,
    BuildSettlement,
    BuyDevCard,
    DiscardResources,
    EndTurn,
    MoveRobber,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayYearOfPlenty,
)
from catan.ai.agent import Agent
from catan.ai.heuristic import RandomAgent
from catan.board import Board
from catan.constants import (
    CITY_COST,
    DEV_CARD_COST,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    MAX_TURNS,
    ROAD_COST,
    SETTLEMENT_COST,
)
from catan.dev_cards import DevCardType
from catan.game import Game, GamePhase
from catan.longest_road import compute_longest_road
from catan.player import Player
from catan.ports import PortType
from catan.resources import TERRAIN_TO_RESOURCE, Resource

# 2d6 probability table for reward shaping (settlement quality)
_REWARD_TOKEN_PROB: dict[int, float] = {
    2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36,
    7: 6 / 36, 8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36, 12: 1 / 36,
}

# Best possible 3-hex vertex production (tokens 6+8+5)
_MAX_VERTEX_PROD = 5 / 36 + 5 / 36 + 4 / 36  # ~0.389

# ------------------------------------------------------------------ #
#  Constants for the flat action space                                 #
# ------------------------------------------------------------------ #

NUM_VERTICES = 54
NUM_EDGES = 72
NUM_HEXES = 19
NUM_PLAYERS = 4
NUM_RESOURCES = 5
NUM_DEV_CARD_TYPES = 5
NUM_PORT_TYPES = 6

# Steal-target encoding: 0 = None, 1–4 = player indices 0–3
_STEAL_SLOTS = 5  # None + 4 players

# Resource ordering (fixed for encoding/decoding)
_RESOURCES = list(Resource)
_RESOURCE_IDX = {r: i for i, r in enumerate(_RESOURCES)}

# Dev card type ordering
_DEV_CARDS = list(DevCardType)
_DEV_CARD_IDX = {d: i for i, d in enumerate(_DEV_CARDS)}

# Port type ordering
_PORT_TYPES = list(PortType)
_PORT_TYPE_IDX = {p: i for i, p in enumerate(_PORT_TYPES)}

# Year of Plenty pairs: unordered (r1, r2) where r1.idx <= r2.idx
_YOP_PAIRS: list[tuple[Resource, Resource]] = []
for _i, _r1 in enumerate(_RESOURCES):
    for _r2 in _RESOURCES[_i:]:
        _YOP_PAIRS.append((_r1, _r2))
assert len(_YOP_PAIRS) == 15

# BankTrade mapping: give × receive (skip same resource) → 20 actions
_BANK_TRADE_PAIRS: list[tuple[Resource, Resource]] = []
for _give in _RESOURCES:
    for _recv in _RESOURCES:
        if _give != _recv:
            _BANK_TRADE_PAIRS.append((_give, _recv))
assert len(_BANK_TRADE_PAIRS) == 20

# ------------------------------------------------------------------ #
#  Action space layout                                                 #
# ------------------------------------------------------------------ #

_OFF_SETTLEMENT = 0  # 54
_OFF_ROAD = _OFF_SETTLEMENT + NUM_VERTICES  # 72
_OFF_CITY = _OFF_ROAD + NUM_EDGES  # 54
_OFF_BANK_TRADE = _OFF_CITY + NUM_VERTICES  # 20
_OFF_BUY_DEV = _OFF_BANK_TRADE + len(_BANK_TRADE_PAIRS)  # 1
_OFF_ROAD_BUILDING = _OFF_BUY_DEV + 1  # 1
_OFF_YOP = _OFF_ROAD_BUILDING + 1  # 15
_OFF_MONOPOLY = _OFF_YOP + len(_YOP_PAIRS)  # 5
_OFF_KNIGHT = _OFF_MONOPOLY + NUM_RESOURCES  # 19 × 5 = 95
_OFF_ROBBER = _OFF_KNIGHT + NUM_HEXES * _STEAL_SLOTS  # 19 × 5 = 95
_OFF_DISCARD = _OFF_ROBBER + NUM_HEXES * _STEAL_SLOTS  # 50
_OFF_END_TURN = _OFF_DISCARD + 50  # 1
TOTAL_ACTIONS = _OFF_END_TURN + 1

# ------------------------------------------------------------------ #
#  Observation space layout                                            #
# ------------------------------------------------------------------ #

# 2d6 probability table (normalised so max = 1.0)
_TOKEN_PROB: dict[int, float] = {
    2: 1 / 36,
    3: 2 / 36,
    4: 3 / 36,
    5: 4 / 36,
    6: 5 / 36,
    7: 6 / 36,
    8: 5 / 36,
    9: 4 / 36,
    10: 3 / 36,
    11: 2 / 36,
    12: 1 / 36,
}

_OBS_HEX_RES = NUM_HEXES * NUM_RESOURCES  # 95
_OBS_HEX_PROB = NUM_HEXES  # 19
_OBS_ROBBER = NUM_HEXES  # 19
_OBS_VERTEX_OWNER = NUM_VERTICES  # 54
_OBS_VERTEX_BLDG = NUM_VERTICES  # 54
_OBS_EDGE_OWNER = NUM_EDGES  # 72
_OBS_VERTEX_PROD = NUM_VERTICES  # 54
_OBS_PLAYER_INCOME = NUM_PLAYERS * NUM_RESOURCES  # 20
_OBS_CAN_AFFORD = 4  # settlement, city, road, dev_card
_OBS_PIECE_COUNTS = NUM_PLAYERS * 3  # 12: settlements, cities, roads per player
_OBS_DIST_AFFORD = 4  # continuous distance to affording each building type
_OBS_LR_LENGTHS = NUM_PLAYERS  # 4: each player's longest road length
_OBS_PLAYED_KNIGHTS = NUM_PLAYERS  # 4: each player's played knight count
_OBS_VP_GAP = 1  # agent VP minus max opponent VP
_OBS_MAX_OPP_VP = 1  # highest opponent VP
_OBS_GAME_PROGRESS = 1  # max VP across all players / 10
_OBS_RES_DIVERSITY = NUM_PLAYERS  # 4: distinct resource types per player
_OBS_PLACEMENT_CTX = 2  # placement round and step
_OBS_VERTEX_RES_DIV = NUM_VERTICES  # 54: distinct resource types per vertex
_OBS_BEST_AVAIL_PROD = 1  # max production among legal placement spots
_OBS_PROD_GAP = NUM_RESOURCES  # 5: binary flags for missing resource types
_OBS_PLAYER_RES = NUM_PLAYERS * NUM_RESOURCES  # 20
_OBS_PLAYER_VP = NUM_PLAYERS  # 4
_OBS_PLAYER_DEV = NUM_PLAYERS * NUM_DEV_CARD_TYPES  # 20
_OBS_PLAYER_KNIGHTS = NUM_PLAYERS  # 4
_OBS_LARGEST_ARMY = NUM_PLAYERS  # 4
_OBS_LONGEST_ROAD = NUM_PLAYERS  # 4
_OBS_PORTS = NUM_PLAYERS * NUM_PORT_TYPES  # 24
_OBS_PHASE = 6  # one-hot over GamePhase values
_OBS_CUR_PLAYER = NUM_PLAYERS  # 4
_OBS_TURN = 1
_OBS_LAST_ROLL = 1
_OBS_DECK_SIZE = 1
_OBS_ROAD_BLD_REM = 1

OBS_SIZE = (
    _OBS_HEX_RES
    + _OBS_HEX_PROB
    + _OBS_ROBBER
    + _OBS_VERTEX_OWNER
    + _OBS_VERTEX_BLDG
    + _OBS_EDGE_OWNER
    + _OBS_VERTEX_PROD
    + _OBS_PLAYER_INCOME
    + _OBS_CAN_AFFORD
    + _OBS_PIECE_COUNTS
    + _OBS_DIST_AFFORD
    + _OBS_LR_LENGTHS
    + _OBS_PLAYED_KNIGHTS
    + _OBS_VP_GAP
    + _OBS_MAX_OPP_VP
    + _OBS_GAME_PROGRESS
    + _OBS_RES_DIVERSITY
    + _OBS_PLACEMENT_CTX
    + _OBS_VERTEX_RES_DIV
    + _OBS_BEST_AVAIL_PROD
    + _OBS_PROD_GAP
    + _OBS_PLAYER_RES
    + _OBS_PLAYER_VP
    + _OBS_PLAYER_DEV
    + _OBS_PLAYER_KNIGHTS
    + _OBS_LARGEST_ARMY
    + _OBS_LONGEST_ROAD
    + _OBS_PORTS
    + _OBS_PHASE
    + _OBS_CUR_PLAYER
    + _OBS_TURN
    + _OBS_LAST_ROLL
    + _OBS_DECK_SIZE
    + _OBS_ROAD_BLD_REM
)

# GamePhase → index for one-hot
_PHASE_IDX = {phase: i for i, phase in enumerate(GamePhase)}

# ------------------------------------------------------------------ #
#  Environment                                                         #
# ------------------------------------------------------------------ #

# Agent seat (training agent is always player 0)
AGENT_SEAT = 0


class CatanEnv(gym.Env):
    """Gymnasium environment for training a single Catan agent.

    The training agent sits at seat 0. Seats 1–3 are controlled by
    ``opponent_agents`` (default: three RandomAgents).  Opponent turns,
    dice rolls, and opponent discard/robber phases are resolved
    automatically so that ``step()`` only returns when the training
    agent must make a decision (or the game ends).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_agents: list[Agent] | None = None,
        seed: int | None = None,
        shuffle_board: bool = True,
    ) -> None:
        super().__init__()

        self.shuffle_board = shuffle_board
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Opponent agents (seats 1–3)
        if opponent_agents is not None:
            assert len(opponent_agents) == 3
            self.opponent_agents = opponent_agents
        else:
            self.opponent_agents = [RandomAgent() for _ in range(3)]

        # Gymnasium spaces
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        # Will be set in reset()
        self.game: Game | None = None
        # Dynamic discard mapping for the current step
        self._discard_actions: list[DiscardResources] = []
        # Reward shaping state
        self._prev_vp: int = 0
        self._prev_longest_road_player: int | None = None
        self._prev_largest_army_player: int | None = None
        self._built_this_turn: bool = False
        self._traded_this_turn: bool = False

    def update_opponent(self, model_path: str, seat: int = 0) -> None:
        """Replace an opponent agent with a new PPOAgent loaded from disk.

        Used by dynamic self-play to periodically refresh the opponent
        with the latest training checkpoint.
        """
        from catan.ai.ppo_agent import PPOAgent

        self.opponent_agents[seat] = PPOAgent(model_path, deterministic=False)

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        game_seed = int(self._rng.integers(0, 2**31))
        import random as _random

        rng = _random.Random(game_seed)
        board = Board.standard(shuffle=self.shuffle_board, rng=rng)
        players = [Player(name=f"P{i}") for i in range(NUM_PLAYERS)]
        self.game = Game(board=board, players=players, rng=rng)

        # Advance past any non-agent phases (placement starts with seat 0,
        # so normally the first legal_actions call is for the agent).
        self._advance_to_agent_decision()

        # Initialise reward shaping baseline
        self._prev_vp = self.game.players[AGENT_SEAT].victory_points
        self._prev_longest_road_player = self.game.longest_road_player
        self._prev_largest_army_player = self.game.largest_army_player
        self._built_this_turn = False
        self._traded_this_turn = False

        return self._encode_obs(), {"action_mask": self.action_masks()}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.game is not None, "Call reset() before step()"

        game = self.game

        # Decode and apply the agent's action
        game_action = self._action_id_to_game_action(action)
        
        # Validate the decoded action is actually legal in the current state
        # If not, fallback to the first legal action as a recovery mechanism
        legal_actions = game.legal_actions()
        if game_action not in legal_actions:
            import warnings
            warnings.warn(
                f"Decoded action {game_action} is not in legal actions. "
                f"Game phase: {game.phase}. Current player: {game.current_player_idx}. "
                f"Using first legal action instead: {legal_actions[0] if legal_actions else 'NONE'}"
            )
            if not legal_actions:
                # No legal actions — treat as a truncated episode to avoid
                # crashing subprocess workers in vectorized training.
                import warnings
                warnings.warn(
                    f"No legal actions available in phase {game.phase}. "
                    "Ending episode as truncated."
                )
                return (
                    self._encode_obs(),
                    -5.0,
                    False,
                    True,
                    {"error": "no_legal_actions"},
                )
            game_action = legal_actions[0]
        
        _was_placement = game.phase == GamePhase.PLACEMENT
        game_over, winner = game.step(game_action)

        if game_over:
            reward = 10.0 if winner == AGENT_SEAT else -5.0
            terminated = True
            truncated = False
            return (
                self._encode_obs(),
                reward,
                terminated,
                truncated,
                {"winner": winner},
            )

        # Advance through non-agent turns / automatic phases
        game_over, winner = self._advance_to_agent_decision()

        if game_over:
            reward = 10.0 if winner == AGENT_SEAT else -5.0
            terminated = True
            truncated = game.turn >= MAX_TURNS
            return (
                self._encode_obs(),
                reward,
                terminated,
                truncated,
                {"winner": winner},
            )

        # Shaped intermediate reward: VP gains + settlement quality + strategic bonuses
        agent = game.players[AGENT_SEAT]
        cur_vp = agent.victory_points
        reward = (cur_vp - self._prev_vp) * 1.0

        # Settlement placement quality: reward based on production value
        if isinstance(game_action, BuildSettlement):
            prod_value = 0.0
            for hex_idx in game.board.topology.vertex_to_hexes[game_action.vertex_id]:
                token = game.board.hexes[hex_idx].token
                if token is not None:
                    prod_value += _REWARD_TOKEN_PROB.get(token, 0.0)
            # Scale: best vertex ~0.42 (6+8+5), normalise to [0.5, 1.5]
            quality_reward = 0.5 + min(prod_value / 0.42, 1.0) * 1.0
            # Double quality reward during placement phase
            if _was_placement:
                quality_reward *= 2.0
            reward += quality_reward

            # Resource diversity bonus: reward gaining new resource types
            new_res_types: set[Resource] = set()
            for hex_idx in game.board.topology.vertex_to_hexes[game_action.vertex_id]:
                terrain = game.board.hexes[hex_idx].terrain
                res = TERRAIN_TO_RESOURCE.get(terrain)
                if res is not None:
                    new_res_types.add(res)
            # Subtract resources already in agent's production portfolio
            existing_types: set[Resource] = set()
            for v_id in agent.settlements + agent.cities:
                if v_id == game_action.vertex_id:
                    continue
                for hex_idx in game.board.topology.vertex_to_hexes[v_id]:
                    terrain = game.board.hexes[hex_idx].terrain
                    res = TERRAIN_TO_RESOURCE.get(terrain)
                    if res is not None:
                        existing_types.add(res)
            reward += 0.3 * len(new_res_types - existing_types)

            # Port bonus: reward settling on port vertices
            for port in game.board.ports:
                if game_action.vertex_id in port.vertices:
                    if port.port_type == PortType.GENERIC:
                        reward += 0.2  # 3:1 port
                    else:
                        reward += 0.3  # 2:1 specialty port
                    break  # a vertex belongs to at most one port
            self._built_this_turn = True

        # City upgrade bonus (supplements the +1.0 VP delta)
        if isinstance(game_action, BuildCity):
            reward += 0.3
            self._built_this_turn = True

        # Knight played bonus (progress toward largest army)
        if isinstance(game_action, PlayKnight):
            reward += 0.15
            self._built_this_turn = True

        # Road/dev card also count as productive actions
        if isinstance(game_action, (BuildRoad, BuyDevCard)):
            self._built_this_turn = True

        # Track bank trades for trade-then-build bonus
        if isinstance(game_action, BankTrade):
            self._traded_this_turn = True

        # Trade-then-build bonus: reward building after trading
        if isinstance(game_action, (BuildSettlement, BuildCity, BuildRoad, BuyDevCard)):
            if self._traded_this_turn:
                reward += 0.05
                self._traded_this_turn = False

        # Longest road gained bonus
        if (
            game.longest_road_player == AGENT_SEAT
            and self._prev_longest_road_player != AGENT_SEAT
        ):
            reward += 1.0

        # Largest army gained bonus
        if (
            game.largest_army_player == AGENT_SEAT
            and self._prev_largest_army_player != AGENT_SEAT
        ):
            reward += 1.0

        # Idle turn penalty: ending turn without building/buying/playing
        if isinstance(game_action, EndTurn) and not self._built_this_turn:
            reward -= 0.05

        # Reset per-turn tracking on EndTurn
        if isinstance(game_action, EndTurn):
            self._built_this_turn = False
            self._traded_this_turn = False

        self._prev_vp = cur_vp
        self._prev_longest_road_player = game.longest_road_player
        self._prev_largest_army_player = game.largest_army_player

        return (
            self._encode_obs(),
            reward,
            False,
            False,
            {"action_mask": self.action_masks()},
        )

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask over the action space (True = legal)."""
        assert self.game is not None
        mask = np.zeros(TOTAL_ACTIONS, dtype=bool)
        legal = self.game.legal_actions()

        # Cache discard actions for decoding
        self._discard_actions = [a for a in legal if isinstance(a, DiscardResources)]

        for action in legal:
            aid = self._game_action_to_id(action)
            if aid is not None:
                mask[aid] = True

        # Safety: an all-zeros mask causes NaN logits in MaskablePPO's masked
        # softmax, crashing training with a Simplex constraint violation.
        # Fall back to EndTurn (a no-op that advances the game) if nothing mapped.
        if not mask.any():
            mask[_OFF_END_TURN] = True
        return mask

    # ------------------------------------------------------------------ #
    #  Observation encoding                                                #
    # ------------------------------------------------------------------ #

    def _encode_obs(self) -> np.ndarray:
        """Encode the full game state as a flat float32 vector in [0, 1]."""
        assert self.game is not None
        game = self.game
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        offset = 0

        # --- Hex resource type (19 × 5 one-hot) ---
        for h in game.board.hexes:
            res = TERRAIN_TO_RESOURCE.get(h.terrain)
            if res is not None:
                obs[offset + _RESOURCE_IDX[res]] = 1.0
            offset += NUM_RESOURCES

        # --- Hex token probability (19) ---
        for h in game.board.hexes:
            if h.token is not None:
                obs[offset] = _TOKEN_PROB.get(h.token, 0.0) / _TOKEN_PROB[6]  # normalise
            offset += 1

        # --- Robber location (19 one-hot) ---
        obs[offset + game.robber_hex] = 1.0
        offset += NUM_HEXES

        # --- Vertex owner (54) ---
        for v in range(NUM_VERTICES):
            owner = game.vertex_owner.get(v)
            if owner is not None:
                obs[offset + v] = (owner + 1) / NUM_PLAYERS  # 0.25, 0.5, 0.75, 1.0
        offset += NUM_VERTICES

        # --- Vertex building type (54) ---
        for v in range(NUM_VERTICES):
            bldg = game.vertex_building.get(v)
            if bldg == "settlement":
                obs[offset + v] = 0.5
            elif bldg == "city":
                obs[offset + v] = 1.0
        offset += NUM_VERTICES

        # --- Edge owner (72) ---
        for e in range(NUM_EDGES):
            owner = game.edge_owner.get(e)
            if owner is not None:
                obs[offset + e] = (owner + 1) / NUM_PLAYERS
        offset += NUM_EDGES

        # --- Vertex production value (54) ---
        # Pre-computed sum of token probabilities for adjacent non-robber hexes.
        # Normalised by best possible 3-hex vertex (tokens 6+8+5 ≈ 0.389).
        for v in range(NUM_VERTICES):
            prod = 0.0
            for hex_idx in game.board.topology.vertex_to_hexes[v]:
                if hex_idx == game.robber_hex:
                    continue
                token = game.board.hexes[hex_idx].token
                if token is not None:
                    prod += _TOKEN_PROB.get(token, 0.0)
            obs[offset + v] = min(prod / _MAX_VERTEX_PROD, 1.0)
        offset += NUM_VERTICES

        # --- Player expected resource income (4 × 5) ---
        # For each player, expected income per resource based on buildings.
        for pidx, player in enumerate(game.players):
            income = [0.0] * NUM_RESOURCES
            for vid in player.settlements:
                for hex_idx in game.board.topology.vertex_to_hexes[vid]:
                    if hex_idx == game.robber_hex:
                        continue
                    h = game.board.hexes[hex_idx]
                    res = TERRAIN_TO_RESOURCE.get(h.terrain)
                    if res is not None and h.token is not None:
                        income[_RESOURCE_IDX[res]] += _TOKEN_PROB.get(h.token, 0.0)
            for vid in player.cities:
                for hex_idx in game.board.topology.vertex_to_hexes[vid]:
                    if hex_idx == game.robber_hex:
                        continue
                    h = game.board.hexes[hex_idx]
                    res = TERRAIN_TO_RESOURCE.get(h.terrain)
                    if res is not None and h.token is not None:
                        income[_RESOURCE_IDX[res]] += _TOKEN_PROB.get(h.token, 0.0) * 2
            for r_idx in range(NUM_RESOURCES):
                obs[offset] = min(income[r_idx] / 0.5, 1.0)
                offset += 1

        # --- Agent affordability flags (4) ---
        agent_player = game.players[AGENT_SEAT]
        obs[offset] = 1.0 if agent_player.can_afford(SETTLEMENT_COST) else 0.0
        obs[offset + 1] = 1.0 if agent_player.can_afford(CITY_COST) else 0.0
        obs[offset + 2] = 1.0 if agent_player.can_afford(ROAD_COST) else 0.0
        obs[offset + 3] = 1.0 if agent_player.can_afford(DEV_CARD_COST) else 0.0
        offset += 4

        # --- Player piece counts (4 × 3 = 12) ---
        for p in game.players:
            obs[offset] = len(p.settlements) / MAX_SETTLEMENTS
            obs[offset + 1] = len(p.cities) / 4.0
            obs[offset + 2] = len(p.roads) / MAX_ROADS
            offset += 3

        # --- Distance to afford (4) ---
        # Continuous signal: 1.0 = can afford, 0.0 = maximally far
        _COSTS = [SETTLEMENT_COST, CITY_COST, ROAD_COST, DEV_CARD_COST]
        for cost in _COSTS:
            total_cost = sum(cost.values())
            missing = 0
            for res, amount in cost.items():
                missing += max(0, amount - agent_player.resources[res])
            obs[offset] = 1.0 - missing / max(total_cost, 1)
            offset += 1

        # --- Longest road lengths (4) ---
        for pidx, p in enumerate(game.players):
            lr = compute_longest_road(
                p.roads, pidx, game.board.topology, game.vertex_owner
            )
            obs[offset] = min(lr / 15.0, 1.0)
            offset += 1

        # --- Played knights (4) ---
        for p in game.players:
            obs[offset] = min(p.played_knights / 8.0, 1.0)
            offset += 1

        # --- VP gap (1) ---
        agent_vp = game.players[AGENT_SEAT].victory_points
        opp_vps = [
            game.players[i].victory_points for i in range(NUM_PLAYERS) if i != AGENT_SEAT
        ]
        max_opp_vp = max(opp_vps) if opp_vps else 0
        obs[offset] = (agent_vp - max_opp_vp + 10) / 20.0
        offset += 1

        # --- Max opponent VP (1) ---
        obs[offset] = min(max_opp_vp / 10.0, 1.0)
        offset += 1

        # --- Game progress (1) ---
        all_vps = [p.victory_points for p in game.players]
        obs[offset] = max(all_vps) / 10.0
        offset += 1

        # --- Resource diversity (4) ---
        for p in game.players:
            distinct = sum(1 for r in _RESOURCES if p.resources[r] > 0)
            obs[offset] = distinct / 5.0
            offset += 1

        # --- Placement context (2) ---
        if game.phase == GamePhase.PLACEMENT:
            obs[offset] = game.placement_round / 5.0
            obs[offset + 1] = game.placement_step / 2.0
        offset += 2

        # --- Vertex resource diversity (54) ---
        for v in range(NUM_VERTICES):
            res_types: set[Resource] = set()
            for hex_idx in game.board.topology.vertex_to_hexes[v]:
                terrain = game.board.hexes[hex_idx].terrain
                res = TERRAIN_TO_RESOURCE.get(terrain)
                if res is not None:
                    res_types.add(res)
            obs[offset + v] = len(res_types) / 3.0
        offset += NUM_VERTICES

        # --- Best available vertex production (1) ---
        if game.phase == GamePhase.PLACEMENT and game.placement_step == 0:
            best_prod = 0.0
            for act in game.legal_actions():
                if isinstance(act, BuildSettlement):
                    prod = 0.0
                    for hex_idx in game.board.topology.vertex_to_hexes[act.vertex_id]:
                        token = game.board.hexes[hex_idx].token
                        if token is not None:
                            prod += _TOKEN_PROB.get(token, 0.0)
                    if prod > best_prod:
                        best_prod = prod
            obs[offset] = min(best_prod / _MAX_VERTEX_PROD, 1.0)
        offset += 1

        # --- Agent resource production gap (5) ---
        agent = game.players[AGENT_SEAT]
        produced_types: set[Resource] = set()
        for v_id in agent.settlements + agent.cities:
            for hex_idx in game.board.topology.vertex_to_hexes[v_id]:
                terrain = game.board.hexes[hex_idx].terrain
                res = TERRAIN_TO_RESOURCE.get(terrain)
                if res is not None:
                    produced_types.add(res)
        for i, r in enumerate(_RESOURCES):
            if r not in produced_types:
                obs[offset + i] = 1.0
        offset += NUM_RESOURCES

        # --- Player resources (4 × 5) ---
        for p in game.players:
            for r in _RESOURCES:
                obs[offset] = min(p.resources[r] / 19.0, 1.0)
                offset += 1

        # --- Player VP (4) ---
        for p in game.players:
            obs[offset] = min(p.victory_points / 10.0, 1.0)
            offset += 1

        # --- Player dev cards (4 × 5) ---
        for p in game.players:
            card_counts = Counter(p.dev_cards)
            for dc in _DEV_CARDS:
                obs[offset] = min(card_counts[dc] / 25.0, 1.0)
                offset += 1

        # --- Player knights played (4) ---
        for p in game.players:
            obs[offset] = min(p.played_knights / 15.0, 1.0)
            offset += 1

        # --- Largest army flag (4) ---
        if game.largest_army_player is not None:
            obs[offset + game.largest_army_player] = 1.0
        offset += NUM_PLAYERS

        # --- Longest road flag (4) ---
        if game.longest_road_player is not None:
            obs[offset + game.longest_road_player] = 1.0
        offset += NUM_PLAYERS

        # --- Port access (4 × 6) ---
        for pidx, p in enumerate(game.players):
            port_types = game._player_port_types(pidx)
            for pt in _PORT_TYPES:
                if pt in port_types:
                    obs[offset] = 1.0
                offset += 1

        # --- Game phase (6 one-hot) ---
        obs[offset + _PHASE_IDX[game.phase]] = 1.0
        offset += 6

        # --- Current player (4 one-hot) ---
        obs[offset + game.current_player_idx] = 1.0
        offset += NUM_PLAYERS

        # --- Scalar features ---
        obs[offset] = game.turn / MAX_TURNS
        offset += 1
        obs[offset] = game.last_roll / 12.0
        offset += 1
        obs[offset] = len(game.dev_card_deck) / 25.0
        offset += 1
        obs[offset] = game.road_building_remaining / 2.0
        offset += 1

        assert offset == OBS_SIZE
        return obs

    # ------------------------------------------------------------------ #
    #  Action encoding / decoding                                          #
    # ------------------------------------------------------------------ #

    def _game_action_to_id(self, action: Action) -> int | None:
        """Map an engine Action to a flat action ID. Returns None if unmappable."""
        match action:
            case BuildSettlement(vertex_id=v):
                return _OFF_SETTLEMENT + v
            case BuildRoad(edge_id=e):
                return _OFF_ROAD + e
            case BuildCity(vertex_id=v):
                return _OFF_CITY + v
            case BankTrade(give=g, receive=r):
                idx = _BANK_TRADE_PAIRS.index((g, r))
                return _OFF_BANK_TRADE + idx
            case BuyDevCard():
                return _OFF_BUY_DEV
            case PlayRoadBuilding():
                return _OFF_ROAD_BUILDING
            case PlayYearOfPlenty(resource1=r1, resource2=r2):
                pair = (r1, r2) if _RESOURCE_IDX[r1] <= _RESOURCE_IDX[r2] else (r2, r1)
                idx = _YOP_PAIRS.index(pair)
                return _OFF_YOP + idx
            case PlayMonopoly(resource=r):
                return _OFF_MONOPOLY + _RESOURCE_IDX[r]
            case PlayKnight(target_hex=h, steal_from=s):
                steal_slot = 0 if s is None else s + 1
                return _OFF_KNIGHT + h * _STEAL_SLOTS + steal_slot
            case MoveRobber(target_hex=h, steal_from=s):
                steal_slot = 0 if s is None else s + 1
                return _OFF_ROBBER + h * _STEAL_SLOTS + steal_slot
            case DiscardResources():
                # Find position in cached discard list
                for i, da in enumerate(self._discard_actions):
                    if da == action:
                        return _OFF_DISCARD + i
                return None
            case EndTurn():
                return _OFF_END_TURN
        return None

    def _action_id_to_game_action(self, action_id: int) -> Action:
        """Decode a flat action ID back to an engine Action."""
        if action_id < _OFF_ROAD:
            return BuildSettlement(vertex_id=action_id - _OFF_SETTLEMENT)
        elif action_id < _OFF_CITY:
            return BuildRoad(edge_id=action_id - _OFF_ROAD)
        elif action_id < _OFF_BANK_TRADE:
            # Decode BuildCity, but validate against current legal actions
            vertex_id = action_id - _OFF_CITY
            # Verify this is actually a legal city action in current game state
            assert self.game is not None
            legal = self.game.legal_actions()
            for action in legal:
                if isinstance(action, BuildCity) and action.vertex_id == vertex_id:
                    return action
            # If not in legal actions, try to construct it anyway as fallback
            # (this will cause apply_action to fail with a clear error)
            return BuildCity(vertex_id=vertex_id)
        elif action_id < _OFF_BUY_DEV:
            idx = action_id - _OFF_BANK_TRADE
            give, recv = _BANK_TRADE_PAIRS[idx]
            return BankTrade(give=give, receive=recv)
        elif action_id == _OFF_BUY_DEV:
            return BuyDevCard()
        elif action_id == _OFF_ROAD_BUILDING:
            return PlayRoadBuilding(edge1=-1, edge2=None)
        elif action_id < _OFF_MONOPOLY:
            idx = action_id - _OFF_YOP
            r1, r2 = _YOP_PAIRS[idx]
            action = PlayYearOfPlenty(resource1=r1, resource2=r2)
            # Validate against current legal actions
            assert self.game is not None
            legal = self.game.legal_actions()
            for legal_action in legal:
                if isinstance(legal_action, PlayYearOfPlenty) and legal_action.resource1 == r1 and legal_action.resource2 == r2:
                    return action
            # Fallback: return anyway (will fail in apply_action with clear error)
            return action
        elif action_id < _OFF_KNIGHT:
            idx = action_id - _OFF_MONOPOLY
            action = PlayMonopoly(resource=_RESOURCES[idx])
            # Validate against current legal actions
            assert self.game is not None
            legal = self.game.legal_actions()
            for legal_action in legal:
                if isinstance(legal_action, PlayMonopoly) and legal_action.resource == _RESOURCES[idx]:
                    return action
            # Fallback: return anyway (will fail in apply_action with clear error)
            return action
        elif action_id < _OFF_ROBBER:
            idx = action_id - _OFF_KNIGHT
            hex_id = idx // _STEAL_SLOTS
            steal_slot = idx % _STEAL_SLOTS
            steal_from = None if steal_slot == 0 else steal_slot - 1
            action = PlayKnight(target_hex=hex_id, steal_from=steal_from)
            # Validate against current legal actions
            assert self.game is not None
            legal = self.game.legal_actions()
            for legal_action in legal:
                if isinstance(legal_action, PlayKnight) and legal_action.target_hex == hex_id and legal_action.steal_from == steal_from:
                    return action
            # Fallback: return anyway (will fail in apply_action with clear error)
            return action
        elif action_id < _OFF_DISCARD:
            idx = action_id - _OFF_ROBBER
            hex_id = idx // _STEAL_SLOTS
            steal_slot = idx % _STEAL_SLOTS
            steal_from = None if steal_slot == 0 else steal_slot - 1
            return MoveRobber(target_hex=hex_id, steal_from=steal_from)
        elif action_id < _OFF_END_TURN:
            idx = action_id - _OFF_DISCARD
            # Recompute discard actions fresh to match current game state
            assert self.game is not None
            legal = self.game.legal_actions()
            discard_actions = [a for a in legal if isinstance(a, DiscardResources)]
            
            if len(discard_actions) == 0:
                # Discard phase has ended (no discard actions available)
                # This can happen when phase changes between mask computation and action application
                # Try to find an EndTurn action instead
                for action in legal:
                    if isinstance(action, EndTurn):
                        return action
                # Fallback: if no EndTurn either, something is very wrong
                raise ValueError(
                    f"Discard action {idx} requested but discard phase has ended. "
                    f"Current game phase: {self.game.phase}, legal actions: {legal}"
                )
            
            if idx < len(discard_actions):
                return discard_actions[idx]
            raise ValueError(
                f"Discard action index {idx} out of range (have {len(discard_actions)} discard actions). "
                f"Game phase: {self.game.phase}"
            )
        elif action_id == _OFF_END_TURN:
            return EndTurn()
        else:
            raise ValueError(f"Invalid action ID: {action_id}")

    # ------------------------------------------------------------------ #
    #  Internal: auto-advance non-agent phases                             #
    # ------------------------------------------------------------------ #

    def _advance_to_agent_decision(self) -> tuple[bool, int | None]:
        """Auto-play until the training agent (seat 0) needs to decide, or game ends.

        Handles: ROLL phase (automatic), opponent turns in all phases,
        opponent discards.

        Returns (game_over, winner).
        """
        assert self.game is not None
        game = self.game

        while True:
            if game.phase == GamePhase.FINISHED:
                winner = game.check_victory()
                return True, winner

            # ROLL phase is automatic — the engine handles it in _start_next_turn
            # so we should never actually be stuck in ROLL phase, but handle it:
            if game.phase == GamePhase.ROLL:
                game._start_next_turn()
                continue

            # Determine who needs to act
            if game.phase == GamePhase.ROBBER_DISCARD:
                acting_player = game.players_to_discard[game._discard_idx]
            else:
                acting_player = game.current_player_idx

            # If it's the training agent's turn, return so step() can collect action
            if acting_player == AGENT_SEAT:
                return False, None

            # Otherwise, auto-play this opponent's action
            legal = game.legal_actions()
            if not legal:
                # Safety: shouldn't happen, but avoid infinite loops
                return True, None

            agent_idx = acting_player - 1  # opponent_agents is 0-indexed for seats 1–3
            opponent = self.opponent_agents[agent_idx]
            action = opponent.choose_action(game, legal)
            game_over, winner = game.step(action)

            if game_over:
                return True, winner
