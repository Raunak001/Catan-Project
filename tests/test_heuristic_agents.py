"""Tests for heuristic baseline agents."""

from __future__ import annotations

import random

import pytest

from catan.actions import (
    BuildCity,
    BuildRoad,
    BuildSettlement,
    BuyDevCard,
    EndTurn,
    PlayKnight,
    PlayRoadBuilding,
)
from catan.ai.heuristic import (
    DevCardBot,
    GreedyAgent,
    LongestRoadBot,
    RandomAgent,
    ResourceHoarder,
)
from catan.game_runner import run_game, run_tournament


# ------------------------------------------------------------------ #
#  Agent name tests                                                    #
# ------------------------------------------------------------------ #


class TestAgentNames:
    def test_random_name(self):
        assert RandomAgent().name() == "Random"

    def test_greedy_name(self):
        assert GreedyAgent().name() == "Greedy"

    def test_longest_road_name(self):
        assert LongestRoadBot().name() == "LongestRoad"

    def test_dev_card_name(self):
        assert DevCardBot().name() == "DevCard"

    def test_resource_hoarder_name(self):
        assert ResourceHoarder().name() == "ResourceHoarder"


# ------------------------------------------------------------------ #
#  Priority behaviour tests                                            #
# ------------------------------------------------------------------ #


class TestGreedyPriority:
    """GreedyAgent should prefer cities > settlements > roads > dev cards."""

    def test_prefers_city_over_settlement(self):
        agent = GreedyAgent(rng=random.Random(0))
        # Mock legal actions: city and settlement both available
        city = BuildCity(vertex_id=10)
        settlement = BuildSettlement(vertex_id=20)
        end = EndTurn()

        # We need a game for production values, but the priority is city > settlement
        # regardless. Use a minimal check: if both city and settlement are legal,
        # the agent picks city.
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        legal = [city, settlement, end]
        action = agent.choose_action(game, legal)
        assert isinstance(action, BuildCity)

    def test_prefers_settlement_over_road(self):
        agent = GreedyAgent(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        settlement = BuildSettlement(vertex_id=10)
        road = BuildRoad(edge_id=5)
        end = EndTurn()
        action = agent.choose_action(game, [settlement, road, end])
        assert isinstance(action, BuildSettlement)

    def test_prefers_road_over_dev_card(self):
        agent = GreedyAgent(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        road = BuildRoad(edge_id=5)
        dev = BuyDevCard()
        end = EndTurn()
        action = agent.choose_action(game, [road, dev, end])
        assert isinstance(action, BuildRoad)

    def test_ends_turn_when_nothing_to_build(self):
        agent = GreedyAgent(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        end = EndTurn()
        action = agent.choose_action(game, [end])
        assert isinstance(action, EndTurn)


class TestLongestRoadPriority:
    """LongestRoadBot should prefer roads > settlements > cities."""

    def test_prefers_road_building_card(self):
        agent = LongestRoadBot(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        rb = PlayRoadBuilding(edge1=0, edge2=1)
        road = BuildRoad(edge_id=5)
        end = EndTurn()
        action = agent.choose_action(game, [rb, road, end])
        assert isinstance(action, PlayRoadBuilding)

    def test_prefers_road_over_settlement(self):
        agent = LongestRoadBot(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        road = BuildRoad(edge_id=5)
        settlement = BuildSettlement(vertex_id=10)
        end = EndTurn()
        action = agent.choose_action(game, [road, settlement, end])
        assert isinstance(action, BuildRoad)

    def test_prefers_settlement_over_city(self):
        agent = LongestRoadBot(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        settlement = BuildSettlement(vertex_id=10)
        city = BuildCity(vertex_id=20)
        end = EndTurn()
        action = agent.choose_action(game, [settlement, city, end])
        assert isinstance(action, BuildSettlement)


class TestDevCardBotPriority:
    """DevCardBot should prefer playing knights > buying dev cards > building."""

    def test_prefers_knight_over_buy(self):
        agent = DevCardBot(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        knight = PlayKnight(target_hex=5, steal_from=None)
        buy = BuyDevCard()
        end = EndTurn()
        action = agent.choose_action(game, [knight, buy, end])
        assert isinstance(action, PlayKnight)

    def test_prefers_buy_over_build(self):
        agent = DevCardBot(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        buy = BuyDevCard()
        city = BuildCity(vertex_id=10)
        end = EndTurn()
        action = agent.choose_action(game, [buy, city, end])
        assert isinstance(action, BuyDevCard)


class TestResourceHoarderPriority:
    """ResourceHoarder should prefer cities > ore/wheat trades > settlements."""

    def test_prefers_city_over_settlement(self):
        agent = ResourceHoarder(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        city = BuildCity(vertex_id=10)
        settlement = BuildSettlement(vertex_id=20)
        end = EndTurn()
        action = agent.choose_action(game, [city, settlement, end])
        assert isinstance(action, BuildCity)


# ------------------------------------------------------------------ #
#  Full game play tests — each agent can complete a game               #
# ------------------------------------------------------------------ #


class TestFullGamePlay:
    """Each agent type should be able to play a complete game without errors."""

    @pytest.mark.parametrize(
        "agent_cls",
        [RandomAgent, GreedyAgent, LongestRoadBot, DevCardBot, ResourceHoarder],
    )
    def test_agent_completes_game(self, agent_cls):
        """Agent can play through a full game without exceptions."""
        agents = [agent_cls(rng=random.Random(i)) for i in range(4)]
        result = run_game(agents, seed=42)
        assert result.turns > 0
        assert all(0 <= vp <= 15 for vp in result.victory_points)

    @pytest.mark.parametrize(
        "agent_cls",
        [GreedyAgent, LongestRoadBot, DevCardBot, ResourceHoarder],
    )
    def test_agent_beats_random_sometimes(self, agent_cls):
        """Heuristic agents should win at least once in 20 games vs randoms."""
        wins = 0
        for seed in range(20):
            agents = [
                agent_cls(rng=random.Random(seed * 10)),
                RandomAgent(rng=random.Random(seed * 10 + 1)),
                RandomAgent(rng=random.Random(seed * 10 + 2)),
                RandomAgent(rng=random.Random(seed * 10 + 3)),
            ]
            result = run_game(agents, seed=seed)
            if result.winner == 0:
                wins += 1
        assert wins >= 1, f"{agent_cls.__name__} didn't win any of 20 games vs Random"


# ------------------------------------------------------------------ #
#  Mixed agent tournaments                                             #
# ------------------------------------------------------------------ #


class TestMixedTournament:
    """Run a small tournament with mixed agent types."""

    def test_mixed_tournament_completes(self):
        agents = [
            GreedyAgent(rng=random.Random(0)),
            LongestRoadBot(rng=random.Random(1)),
            DevCardBot(rng=random.Random(2)),
            ResourceHoarder(rng=random.Random(3)),
        ]
        result = run_tournament(agents, n_games=5, base_seed=42)
        assert result.total_games == 5
        assert sum(result.wins) + result.draws <= 5
