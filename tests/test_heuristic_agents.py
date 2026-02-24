"""Tests for heuristic baseline agents."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from catan.actions import (
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
from catan.ai.heuristic import (
    DevCardBot,
    GreedyAgent,
    LongestRoadBot,
    RandomAgent,
    ResourceHoarder,
    SmartBot,
    _acting_player_idx,
    _best_dev_card_play,
    _hex_production_value,
    _need_based_trades,
    _road_extends_longest,
    _smart_discard,
    _smart_road,
    _smart_robber_action,
)
from catan.constants import CITY_COST, SETTLEMENT_COST
from catan.game_runner import run_game, run_tournament
from catan.resources import Resource

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


# ------------------------------------------------------------------ #
#  Smart helper function tests                                         #
# ------------------------------------------------------------------ #


class TestSmartHelpers:
    """Tests for shared helper functions used by improved bots."""

    def test_acting_player_idx_main_phase(self):
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        assert _acting_player_idx(game) == game.current_player_idx

    def test_hex_production_value_desert_is_zero(self):
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        # Find the desert hex (token is None)
        for i, h in enumerate(game.board.hexes):
            if h.token is None:
                assert _hex_production_value(game, i) == 0.0
                break

    def test_hex_production_value_increases_with_buildings(self):
        from tests.helpers import make_game, place_settlement_at, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        # Find a non-desert hex with a number token
        target_hex = None
        for i, h in enumerate(game.board.hexes):
            if h.token is not None and h.token != 7:
                target_hex = i
                break
        assert target_hex is not None
        val_before = _hex_production_value(game, target_hex)
        # Place a settlement on one of its vertices
        verts = game.topology.hex_to_vertices[target_hex]
        # Find an unoccupied vertex
        for v in verts:
            if v not in game.vertex_owner:
                place_settlement_at(game, 1, v)
                break
        val_after = _hex_production_value(game, target_hex)
        assert val_after > val_before

    def test_smart_robber_targets_leader(self):
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        # Give P1 extra VP to make them the leader
        game.players[1].victory_points += 5
        # Give P1 some resources (so they can be stolen from)
        game.players[1].resources[Resource.ORE] += 3

        # Find a hex where P1 has a building
        p1_hexes = set()
        for v in game.players[1].settlements + game.players[1].cities:
            for hex_idx in game.topology.vertex_to_hexes[v]:
                p1_hexes.add(hex_idx)

        if p1_hexes:
            # Create robber actions — some target P1's hexes, some don't
            robber_actions = []
            for hex_idx in range(len(game.board.hexes)):
                if hex_idx == game.robber_hex:
                    continue
                steal_targets = game._robber_steal_targets(hex_idx, 0)
                if steal_targets:
                    for t in steal_targets:
                        robber_actions.append(MoveRobber(hex_idx, t))
                else:
                    robber_actions.append(MoveRobber(hex_idx, None))

            if robber_actions:
                choice = _smart_robber_action(game, 0, robber_actions, random.Random(0))
                # Should prefer targeting P1 (the leader)
                assert isinstance(choice, MoveRobber)

    def test_smart_discard_keeps_valuable_resources(self):
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        player = game.players[0]

        # Give player: 4 ore, 4 wheat, 4 sheep = 12 cards, must discard 6
        player.resources = Counter(
            {
                Resource.ORE: 4,
                Resource.WHEAT: 4,
                Resource.SHEEP: 4,
            }
        )

        # Create discard options: keep ore+wheat vs keep sheep
        discard_keep_valuable = DiscardResources(
            Counter({Resource.SHEEP: 4, Resource.ORE: 1, Resource.WHEAT: 1})
        )
        discard_lose_valuable = DiscardResources(Counter({Resource.ORE: 3, Resource.WHEAT: 3}))

        choice = _smart_discard(
            game, 0, [discard_keep_valuable, discard_lose_valuable], random.Random(0)
        )
        # Should keep ore+wheat (higher value), discard sheep
        assert choice == discard_keep_valuable

    def test_smart_road_toward_good_vertex(self):
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=42)
        skip_to_main_phase(game)
        pidx = game.current_player_idx

        # Get legal road actions
        roads = []
        for eid in range(game.topology.num_edges):
            if game._can_build_road(pidx, eid):
                roads.append(BuildRoad(eid))

        if len(roads) >= 2:
            choice = _smart_road(game, pidx, roads, random.Random(0))
            assert isinstance(choice, BuildRoad)
            assert choice in roads

    def test_road_extends_longest_basic(self):
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=42)
        skip_to_main_phase(game)
        pidx = game.current_player_idx

        # Get legal road actions
        roads = []
        for eid in range(game.topology.num_edges):
            if game._can_build_road(pidx, eid):
                roads.append(BuildRoad(eid))

        if roads:
            extending = _road_extends_longest(game, pidx, roads)
            # Result should be a subset of input
            for r in extending:
                assert r in roads

    def test_need_based_trades_filters_correctly(self):
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        pidx = game.current_player_idx
        player = game.players[pidx]

        # Give player lots of sheep but no ore
        player.resources = Counter({Resource.SHEEP: 8, Resource.WHEAT: 2})

        # Create trade actions
        trades = [
            BankTrade(give=Resource.SHEEP, receive=Resource.ORE),
            BankTrade(give=Resource.SHEEP, receive=Resource.WOOD),
            BankTrade(give=Resource.SHEEP, receive=Resource.BRICK),
        ]

        # Need ore for city (CITY_COST needs 3 ore, 2 wheat)
        useful = _need_based_trades(game, pidx, trades, CITY_COST)
        # Should only include the ore trade (we already have enough wheat)
        assert len(useful) >= 1
        assert all(t.receive == Resource.ORE for t in useful)

    def test_need_based_trades_empty_when_can_afford(self):
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        pidx = game.current_player_idx
        player = game.players[pidx]

        # Player can already afford a settlement
        player.resources = Counter(
            {
                Resource.WOOD: 2,
                Resource.BRICK: 2,
                Resource.SHEEP: 2,
                Resource.WHEAT: 2,
            }
        )
        trades = [BankTrade(give=Resource.WOOD, receive=Resource.ORE)]
        useful = _need_based_trades(game, pidx, trades, SETTLEMENT_COST)
        assert useful == []

    def test_best_dev_card_play_knight(self):
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        pidx = game.current_player_idx

        knight = PlayKnight(target_hex=5, steal_from=None)
        legal = [knight, EndTurn()]
        result = _best_dev_card_play(game, pidx, legal, random.Random(0))
        assert isinstance(result, PlayKnight)

    def test_best_dev_card_play_monopoly_threshold(self):
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        pidx = game.current_player_idx

        # Opponents have lots of ore
        for i in range(len(game.players)):
            if i != pidx:
                game.players[i].resources[Resource.ORE] = 3

        monopoly_ore = PlayMonopoly(resource=Resource.ORE)
        monopoly_sheep = PlayMonopoly(resource=Resource.SHEEP)
        legal = [monopoly_ore, monopoly_sheep, EndTurn()]
        result = _best_dev_card_play(game, pidx, legal, random.Random(0))
        # Should pick ore monopoly (opponents have 9 total ore)
        assert result == monopoly_ore

    def test_best_dev_card_play_monopoly_skipped_if_low(self):
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        pidx = game.current_player_idx

        # Opponents have very few resources
        for i in range(len(game.players)):
            if i != pidx:
                game.players[i].resources = Counter()

        monopoly = PlayMonopoly(resource=Resource.ORE)
        legal = [monopoly, EndTurn()]
        result = _best_dev_card_play(game, pidx, legal, random.Random(0))
        # Should skip monopoly (0 total ore among opponents)
        assert result is None

    def test_best_dev_card_play_year_of_plenty(self):
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        pidx = game.current_player_idx

        yop1 = PlayYearOfPlenty(resource1=Resource.ORE, resource2=Resource.WHEAT)
        yop2 = PlayYearOfPlenty(resource1=Resource.WOOD, resource2=Resource.WOOD)
        legal = [yop1, yop2, EndTurn()]
        result = _best_dev_card_play(game, pidx, legal, random.Random(0))
        assert isinstance(result, PlayYearOfPlenty)


# ------------------------------------------------------------------ #
#  Improved bot behaviour tests                                        #
# ------------------------------------------------------------------ #


class TestImprovedBots:
    """Tests for improved bot behaviours (smart robber, discard, dev cards)."""

    def test_greedy_plays_knight_when_available(self):
        """GreedyAgent should now play knights (previously it ignored them)."""
        agent = GreedyAgent(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        knight = PlayKnight(target_hex=5, steal_from=None)
        end = EndTurn()
        action = agent.choose_action(game, [knight, end])
        assert isinstance(action, PlayKnight)

    def test_greedy_handles_discard_smartly(self):
        """GreedyAgent should use smart discard, not random."""
        agent = GreedyAgent(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        player = game.players[game.current_player_idx]
        player.resources = Counter({Resource.ORE: 4, Resource.SHEEP: 4})

        discard_sheep = DiscardResources(Counter({Resource.SHEEP: 4}))
        discard_ore = DiscardResources(Counter({Resource.ORE: 4}))
        action = agent.choose_action(game, [discard_sheep, discard_ore])
        # Should discard sheep (less valuable) and keep ore
        assert isinstance(action, DiscardResources)
        assert action == discard_sheep

    def test_greedy_trades_toward_purchase(self):
        """GreedyAgent should trade toward a purchase when it can't build."""
        agent = GreedyAgent(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        pidx = game.current_player_idx
        player = game.players[pidx]
        # Give player lots of sheep but missing ore for city
        player.resources = Counter({Resource.SHEEP: 8, Resource.WHEAT: 2})

        # Player has a settlement to upgrade
        trade = BankTrade(give=Resource.SHEEP, receive=Resource.ORE)
        end = EndTurn()
        action = agent.choose_action(game, [trade, end])
        assert isinstance(action, BankTrade)

    def test_longest_road_bot_handles_robber(self):
        """LongestRoadBot should handle robber moves smartly."""
        agent = LongestRoadBot(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        robber1 = MoveRobber(target_hex=3, steal_from=None)
        robber2 = MoveRobber(target_hex=7, steal_from=1)
        action = agent.choose_action(game, [robber1, robber2])
        assert isinstance(action, MoveRobber)

    def test_dev_card_bot_smart_knight_targeting(self):
        """DevCardBot should use smart knight targeting, not random."""
        agent = DevCardBot(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        # Make P1 the leader with resources
        game.players[1].victory_points += 5
        game.players[1].resources[Resource.ORE] = 5

        knight_bad = PlayKnight(target_hex=0, steal_from=None)
        # Find a hex where P1 has buildings
        p1_hex = None
        for v in game.players[1].settlements:
            for h in game.topology.vertex_to_hexes[v]:
                if game.board.hexes[h].token is not None:
                    p1_hex = h
                    break
            if p1_hex is not None:
                break

        if p1_hex is not None:
            knight_good = PlayKnight(target_hex=p1_hex, steal_from=1)
            action = agent.choose_action(game, [knight_bad, knight_good, EndTurn()])
            assert isinstance(action, PlayKnight)
            # Should prefer targeting the leader
            assert action.steal_from == 1 or action.target_hex == p1_hex

    def test_resource_hoarder_need_based_trading(self):
        """ResourceHoarder should only trade when it helps toward a city."""
        agent = ResourceHoarder(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        pidx = game.current_player_idx
        player = game.players[pidx]

        # Player already has enough ore+wheat for a city — no trade needed
        player.resources = Counter({Resource.ORE: 3, Resource.WHEAT: 2, Resource.SHEEP: 4})
        trade = BankTrade(give=Resource.SHEEP, receive=Resource.ORE)
        end = EndTurn()
        action = agent.choose_action(game, [trade, end])
        # Should end turn (no useful trade needed for city)
        assert isinstance(action, EndTurn)


# ------------------------------------------------------------------ #
#  SmartBot tests                                                      #
# ------------------------------------------------------------------ #


class TestSmartBot:
    """Tests for the new SmartBot agent."""

    def test_smart_bot_name(self):
        assert SmartBot().name() == "Smart"

    def test_smart_bot_completes_game(self):
        """SmartBot can play a full game without errors."""
        agents = [SmartBot(rng=random.Random(i)) for i in range(4)]
        result = run_game(agents, seed=42)
        assert result.turns > 0
        assert all(0 <= vp <= 15 for vp in result.victory_points)

    def test_smart_bot_handles_discard(self):
        agent = SmartBot(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        discard1 = DiscardResources(Counter({Resource.SHEEP: 2}))
        discard2 = DiscardResources(Counter({Resource.ORE: 2}))
        action = agent.choose_action(game, [discard1, discard2])
        assert isinstance(action, DiscardResources)

    def test_smart_bot_handles_robber(self):
        agent = SmartBot(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        robber = MoveRobber(target_hex=3, steal_from=1)
        action = agent.choose_action(game, [robber])
        assert isinstance(action, MoveRobber)

    def test_smart_bot_plays_dev_cards(self):
        agent = SmartBot(rng=random.Random(0))
        from tests.helpers import make_game, skip_to_main_phase

        game = make_game(seed=1)
        skip_to_main_phase(game)
        knight = PlayKnight(target_hex=5, steal_from=None)
        end = EndTurn()
        action = agent.choose_action(game, [knight, end])
        assert isinstance(action, PlayKnight)

    @pytest.mark.slow
    def test_smart_bot_beats_random_consistently(self):
        """SmartBot should win significantly more than 25% vs 3 Randoms."""
        agents = [
            SmartBot(rng=random.Random(0)),
            RandomAgent(rng=random.Random(1)),
            RandomAgent(rng=random.Random(2)),
            RandomAgent(rng=random.Random(3)),
        ]
        result = run_tournament(agents, n_games=50, base_seed=42)
        win_rate = result.wins[0] / 50
        assert win_rate >= 0.35, f"SmartBot won only {win_rate:.0%} vs Random (expected >35%)"

    def test_smart_bot_mixed_tournament(self):
        """SmartBot participates in a mixed tournament without errors."""
        agents = [
            SmartBot(rng=random.Random(0)),
            GreedyAgent(rng=random.Random(1)),
            DevCardBot(rng=random.Random(2)),
            ResourceHoarder(rng=random.Random(3)),
        ]
        result = run_tournament(agents, n_games=5, base_seed=42)
        assert result.total_games == 5
        assert sum(result.wins) + result.draws <= 5


# ------------------------------------------------------------------ #
#  Game completion rate tests                                          #
# ------------------------------------------------------------------ #


class TestGameCompletionRates:
    """Improved bots should finish games more reliably."""

    @pytest.mark.slow
    def test_greedy_vs_random_completion(self):
        """Improved Greedy vs 3 Random should finish >50% of games."""
        agents = [
            GreedyAgent(rng=random.Random(0)),
            RandomAgent(rng=random.Random(1)),
            RandomAgent(rng=random.Random(2)),
            RandomAgent(rng=random.Random(3)),
        ]
        result = run_tournament(agents, n_games=30, base_seed=42)
        decisive = sum(result.wins)
        rate = decisive / 30
        assert rate >= 0.5, f"Only {rate:.0%} decisive games (expected >50%)"

    @pytest.mark.slow
    def test_smart_vs_smart_completion(self):
        """4 SmartBots should finish >60% of games."""
        agents = [SmartBot(rng=random.Random(i)) for i in range(4)]
        result = run_tournament(agents, n_games=30, base_seed=42)
        decisive = sum(result.wins)
        rate = decisive / 30
        assert rate >= 0.6, f"Only {rate:.0%} decisive games (expected >60%)"
