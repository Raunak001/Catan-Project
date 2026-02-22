"""Tests for development card mechanics."""

from collections import Counter

from catan.actions import (
    BuyDevCard,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayYearOfPlenty,
)
from catan.constants import DEV_CARD_COST
from catan.dev_cards import DECK_COMPOSITION, DevCardType, make_deck
from catan.game import GamePhase
from catan.resources import Resource

from .helpers import (
    give_dev_card,
    give_resources,
    make_game,
    skip_to_main_phase,
)


class TestDeckComposition:
    def test_deck_has_25_cards(self):
        deck = make_deck()
        assert len(deck) == 25

    def test_deck_has_correct_counts(self):
        import random

        deck = make_deck(random.Random(0))
        counts = Counter(deck)
        assert counts[DevCardType.KNIGHT] == 14
        assert counts[DevCardType.VICTORY_POINT] == 5
        assert counts[DevCardType.ROAD_BUILDING] == 2
        assert counts[DevCardType.YEAR_OF_PLENTY] == 2
        assert counts[DevCardType.MONOPOLY] == 2


class TestBuyDevCard:
    def test_buy_requires_sheep_wheat_ore(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        player.resources = Counter()
        actions = game.legal_actions()
        assert not any(isinstance(a, BuyDevCard) for a in actions)

    def test_buy_with_resources_available(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_resources(player, {Resource.SHEEP: 1, Resource.WHEAT: 1, Resource.ORE: 1})
        actions = game.legal_actions()
        assert any(isinstance(a, BuyDevCard) for a in actions)

    def test_buy_deducts_resources(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_resources(player, {Resource.SHEEP: 1, Resource.WHEAT: 1, Resource.ORE: 1})
        game.apply_action(BuyDevCard())
        assert player.resources[Resource.SHEEP] == 0
        assert player.resources[Resource.WHEAT] == 0
        assert player.resources[Resource.ORE] == 0

    def test_bought_card_goes_to_new_dev_cards(self):
        """Cards bought this turn go to new_dev_cards, not dev_cards."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_resources(player, {Resource.SHEEP: 1, Resource.WHEAT: 1, Resource.ORE: 1})
        before_dev = len(player.dev_cards)
        before_new = len(player.new_dev_cards)
        game.apply_action(BuyDevCard())
        assert len(player.new_dev_cards) == before_new + 1
        # dev_cards unchanged (card is in new_dev_cards)
        assert len(player.dev_cards) == before_dev

    def test_cannot_play_card_bought_this_turn(self):
        """Cards in new_dev_cards should not be playable."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        # Manually put a knight in new_dev_cards (simulating just-bought)
        player.new_dev_cards.append(DevCardType.KNIGHT)
        actions = game.legal_actions()
        assert not any(isinstance(a, PlayKnight) for a in actions)

    def test_new_cards_move_to_hand_on_end_turn(self):
        """At end of turn, new_dev_cards should transfer to dev_cards."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        player.new_dev_cards.append(DevCardType.KNIGHT)
        game._end_current_turn()
        assert DevCardType.KNIGHT in player.dev_cards
        assert len(player.new_dev_cards) == 0

    def test_cannot_buy_when_deck_empty(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_resources(player, {Resource.SHEEP: 5, Resource.WHEAT: 5, Resource.ORE: 5})
        game.dev_card_deck.clear()
        actions = game.legal_actions()
        assert not any(isinstance(a, BuyDevCard) for a in actions)

    def test_vp_card_immediately_adds_point(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_resources(player, {Resource.SHEEP: 1, Resource.WHEAT: 1, Resource.ORE: 1})
        vp_before = player.victory_points
        # Stack the deck so next card is VP
        game.dev_card_deck.append(DevCardType.VICTORY_POINT)
        game.apply_action(BuyDevCard())
        assert player.victory_points == vp_before + 1


class TestPlayKnight:
    def test_knight_moves_robber(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_dev_card(player, DevCardType.KNIGHT)
        old_robber = game.robber_hex
        target = 0 if old_robber != 0 else 1
        game.apply_action(PlayKnight(target, None))
        assert game.robber_hex == target

    def test_knight_increments_played_knights(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_dev_card(player, DevCardType.KNIGHT)
        assert player.played_knights == 0
        target = 0 if game.robber_hex != 0 else 1
        game.apply_action(PlayKnight(target, None))
        assert player.played_knights == 1

    def test_knight_removes_from_hand(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_dev_card(player, DevCardType.KNIGHT)
        target = 0 if game.robber_hex != 0 else 1
        game.apply_action(PlayKnight(target, None))
        assert DevCardType.KNIGHT not in player.dev_cards

    def test_only_one_dev_card_per_turn(self):
        """After playing a dev card, cannot play another."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_dev_card(player, DevCardType.KNIGHT)
        give_dev_card(player, DevCardType.KNIGHT)
        target = 0 if game.robber_hex != 0 else 1
        game.apply_action(PlayKnight(target, None))
        assert player.dev_card_played_this_turn
        actions = game.legal_actions()
        assert not any(isinstance(a, PlayKnight) for a in actions)


class TestLargestArmy:
    def test_three_knights_grants_largest_army(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        pidx = game.current_player_idx
        vp_before = player.victory_points

        for i in range(3):
            give_dev_card(player, DevCardType.KNIGHT)
            player.dev_card_played_this_turn = False
            target = (game.robber_hex + 1) % len(game.board.hexes)
            game.apply_action(PlayKnight(target, None))

        assert game.largest_army_player == pidx
        assert player.victory_points == vp_before + 2

    def test_largest_army_stolen_by_more_knights(self):
        game = make_game()
        skip_to_main_phase(game)
        p0 = game.players[0]
        p1 = game.players[1]

        # Give P0 largest army (3 knights)
        game.current_player_idx = 0
        for _ in range(3):
            give_dev_card(p0, DevCardType.KNIGHT)
            p0.dev_card_played_this_turn = False
            target = (game.robber_hex + 1) % len(game.board.hexes)
            game.apply_action(PlayKnight(target, None))
        assert game.largest_army_player == 0
        vp0_with_army = p0.victory_points

        # Now P1 plays 4 knights
        game.current_player_idx = 1
        for _ in range(4):
            give_dev_card(p1, DevCardType.KNIGHT)
            p1.dev_card_played_this_turn = False
            target = (game.robber_hex + 1) % len(game.board.hexes)
            game.apply_action(PlayKnight(target, None))
        assert game.largest_army_player == 1
        # P0 lost 2 VP
        assert p0.victory_points == vp0_with_army - 2


class TestPlayYearOfPlenty:
    def test_grants_two_resources(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_dev_card(player, DevCardType.YEAR_OF_PLENTY)
        before_wood = player.resources[Resource.WOOD]
        before_ore = player.resources[Resource.ORE]
        game.apply_action(PlayYearOfPlenty(Resource.WOOD, Resource.ORE))
        assert player.resources[Resource.WOOD] == before_wood + 1
        assert player.resources[Resource.ORE] == before_ore + 1

    def test_can_pick_same_resource_twice(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_dev_card(player, DevCardType.YEAR_OF_PLENTY)
        before = player.resources[Resource.BRICK]
        game.apply_action(PlayYearOfPlenty(Resource.BRICK, Resource.BRICK))
        assert player.resources[Resource.BRICK] == before + 2


class TestPlayMonopoly:
    def test_steals_all_of_one_resource(self):
        game = make_game()
        skip_to_main_phase(game)
        pidx = game.current_player_idx
        player = game.current_player
        give_dev_card(player, DevCardType.MONOPOLY)

        # Give opponents wheat
        for i, p in enumerate(game.players):
            if i != pidx:
                give_resources(p, {Resource.WHEAT: 3})

        total_wheat_from_others = sum(
            p.resources[Resource.WHEAT]
            for i, p in enumerate(game.players)
            if i != pidx
        )
        before = player.resources[Resource.WHEAT]
        game.apply_action(PlayMonopoly(Resource.WHEAT))

        assert player.resources[Resource.WHEAT] == before + total_wheat_from_others
        for i, p in enumerate(game.players):
            if i != pidx:
                assert p.resources[Resource.WHEAT] == 0


class TestPlayRoadBuilding:
    def test_road_building_sets_remaining_to_2(self):
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_dev_card(player, DevCardType.ROAD_BUILDING)
        game.apply_action(PlayRoadBuilding(edge1=-1, edge2=None))
        assert game.road_building_remaining == 2

    def test_road_building_places_free_roads(self):
        """After playing road building, next actions should be free road placements."""
        game = make_game()
        skip_to_main_phase(game)
        player = game.current_player
        give_dev_card(player, DevCardType.ROAD_BUILDING)
        roads_before = len(player.roads)
        resources_before = dict(player.resources)

        game.apply_action(PlayRoadBuilding(edge1=-1, edge2=None))

        from catan.actions import BuildRoad

        actions = game.legal_actions()
        assert all(isinstance(a, BuildRoad) for a in actions)
        # Place first road
        if actions:
            game.apply_action(actions[0])
            assert len(player.roads) == roads_before + 1
            assert game.road_building_remaining == 1
            # Resources unchanged (free roads)
            assert dict(player.resources) == resources_before
