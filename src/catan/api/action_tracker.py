"""Track per-game action history and compute aggregate statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from catan.actions import Action, BankTrade, DiscardResources
from catan.dev_cards import DevCardType
from catan.game import Game
from catan.resources import Resource


@dataclass
class ActionRecord:
    """A single recorded action with context captured before apply_action."""

    turn: int
    phase: str
    player_idx: int
    action_type: str
    action_details: dict
    vp_snapshot: list[int]


@dataclass
class ActionStats:
    """Aggregate statistics derived from action records."""

    total_actions: int
    action_type_counts: dict[str, int]
    per_player: dict[int, dict[str, int]]
    phase_distribution: dict[str, int]
    trades_per_player: dict[int, int]
    dev_cards_played: dict[str, int]
    robber_moves: int
    vp_timeline: list[dict]


def _serialize_action_details(action: Action) -> dict:
    """Extract action fields as a JSON-safe dict."""
    match action:
        case BankTrade(give=give, receive=recv):
            return {"give": give.value, "receive": recv.value}
        case DiscardResources(resources=res):
            return {"resources": {r.value: c for r, c in res.items() if c > 0}}
        case _:
            details: dict = {}
            for k, v in action.__dict__.items():
                if isinstance(v, Resource):
                    details[k] = v.value
                elif isinstance(v, DevCardType):
                    details[k] = v.value
                elif isinstance(v, Counter):
                    details[k] = {r.value: c for r, c in v.items() if c > 0}
                else:
                    details[k] = v
            return details


_DEV_CARD_ACTION_TYPES = {"PlayKnight", "PlayRoadBuilding", "PlayYearOfPlenty", "PlayMonopoly"}


class ActionTracker:
    """Records actions during a game and computes statistics."""

    def __init__(self) -> None:
        self.records: list[ActionRecord] = []

    def record(self, game: Game, player_idx: int, action: Action) -> None:
        """Record an action. Must be called BEFORE game.apply_action()."""
        self.records.append(
            ActionRecord(
                turn=game.turn,
                phase=game.phase.value,
                player_idx=player_idx,
                action_type=type(action).__name__,
                action_details=_serialize_action_details(action),
                vp_snapshot=[p.victory_points for p in game.players],
            )
        )

    def get_stats(self) -> ActionStats:
        """Compute aggregate statistics from recorded actions."""
        type_counts: Counter[str] = Counter()
        per_player: dict[int, Counter[str]] = {}
        phase_dist: Counter[str] = Counter()
        trades: Counter[int] = Counter()
        dev_cards: Counter[str] = Counter()
        robber_moves = 0
        vp_timeline: list[dict] = []
        seen_turns: set[int] = set()

        for rec in self.records:
            type_counts[rec.action_type] += 1
            phase_dist[rec.phase] += 1

            if rec.player_idx not in per_player:
                per_player[rec.player_idx] = Counter()
            per_player[rec.player_idx][rec.action_type] += 1

            if rec.action_type == "BankTrade":
                trades[rec.player_idx] += 1

            if rec.action_type in _DEV_CARD_ACTION_TYPES:
                dev_cards[rec.action_type] += 1

            if rec.action_type in ("MoveRobber", "PlayKnight"):
                robber_moves += 1

            if rec.turn not in seen_turns:
                seen_turns.add(rec.turn)
                vp_timeline.append({"turn": rec.turn, "vps": rec.vp_snapshot})

        return ActionStats(
            total_actions=len(self.records),
            action_type_counts=dict(type_counts),
            per_player={k: dict(v) for k, v in per_player.items()},
            phase_distribution=dict(phase_dist),
            trades_per_player=dict(trades),
            dev_cards_played=dict(dev_cards),
            robber_moves=robber_moves,
            vp_timeline=vp_timeline,
        )
