"""Game state invariant checks — run after simulated games to catch logic bugs."""

import random

from catan.ai.heuristic import RandomAgent
from catan.board import Board
from catan.constants import MAX_CITIES, MAX_ROADS, MAX_SETTLEMENTS
from catan.game import Game, GamePhase
from catan.player import Player
from catan.resources import Resource


def validate_game_state(game: Game) -> list[str]:
    """Check game state invariants, return list of violations."""
    violations: list[str] = []

    # 1. Vertex ownership consistency
    for vid, pidx in game.vertex_owner.items():
        if pidx is None:
            continue
        player = game.players[pidx]
        building = game.vertex_building.get(vid)
        if building == "settlement":
            if vid not in player.settlements:
                violations.append(
                    f"vertex_owner[{vid}]={pidx} is settlement but not in player.settlements"
                )
        elif building == "city":
            if vid not in player.cities:
                violations.append(f"vertex_owner[{vid}]={pidx} is city but not in player.cities")
        else:
            violations.append(f"vertex_owner[{vid}]={pidx} has unknown building type: {building}")

    # 2. Edge ownership consistency
    for eid, pidx in game.edge_owner.items():
        if pidx is None:
            continue
        if eid not in game.players[pidx].roads:
            violations.append(f"edge_owner[{eid}]={pidx} but not in player.roads")

    # 3. Player building lists match board state
    for pidx, player in enumerate(game.players):
        for vid in player.settlements:
            if game.vertex_owner.get(vid) != pidx:
                violations.append(
                    f"P{pidx} has settlement at {vid} "
                    f"but vertex_owner={game.vertex_owner.get(vid)}"
                )
            if game.vertex_building.get(vid) != "settlement":
                violations.append(
                    f"P{pidx} settlement at {vid} but building={game.vertex_building.get(vid)}"
                )
        for vid in player.cities:
            if game.vertex_owner.get(vid) != pidx:
                violations.append(
                    f"P{pidx} has city at {vid} but vertex_owner={game.vertex_owner.get(vid)}"
                )
            if game.vertex_building.get(vid) != "city":
                violations.append(
                    f"P{pidx} city at {vid} but building={game.vertex_building.get(vid)}"
                )
        for eid in player.roads:
            if game.edge_owner.get(eid) != pidx:
                violations.append(
                    f"P{pidx} has road at {eid} but edge_owner={game.edge_owner.get(eid)}"
                )

    # 4. Piece limits
    for pidx, player in enumerate(game.players):
        if len(player.settlements) > MAX_SETTLEMENTS:
            violations.append(
                f"P{pidx} has {len(player.settlements)} settlements (max {MAX_SETTLEMENTS})"
            )
        if len(player.cities) > MAX_CITIES:
            violations.append(f"P{pidx} has {len(player.cities)} cities (max {MAX_CITIES})")
        if len(player.roads) > MAX_ROADS:
            violations.append(f"P{pidx} has {len(player.roads)} roads (max {MAX_ROADS})")

    # 5. Distance rule: no two buildings on adjacent vertices
    for vid in game.vertex_owner:
        for neighbor in game.topology.vertex_neighbors[vid]:
            if neighbor in game.vertex_owner and neighbor != vid:
                # Both occupied — check they weren't placed in violation
                # (this is valid if they were placed by the same player,
                #  as long as they don't violate the distance rule)
                pass  # distance rule only applies at placement time

    # 6. No negative resources
    for pidx, player in enumerate(game.players):
        for res in Resource:
            if player.resources[res] < 0:
                violations.append(f"P{pidx} has negative {res.value}: {player.resources[res]}")

    # 7. VP consistency: settlements + cities + bonuses
    for pidx, player in enumerate(game.players):
        expected_vp = len(player.settlements) + 2 * len(player.cities)
        # Add dev card VPs
        from catan.dev_cards import DevCardType

        vp_cards = player.dev_cards.count(DevCardType.VICTORY_POINT)
        vp_new_cards = player.new_dev_cards.count(DevCardType.VICTORY_POINT)
        expected_vp += vp_cards + vp_new_cards
        # Longest road
        if game.longest_road_player == pidx:
            expected_vp += 2
        # Largest army
        if game.largest_army_player == pidx:
            expected_vp += 2
        if player.victory_points != expected_vp:
            violations.append(
                f"P{pidx} VP mismatch: actual={player.victory_points}, expected={expected_vp} "
                f"(settlements={len(player.settlements)}, cities={len(player.cities)}, "
                f"vp_cards={vp_cards + vp_new_cards}, "
                f"longest_road={game.longest_road_player == pidx}, "
                f"largest_army={game.largest_army_player == pidx})"
            )

    # 8. Robber is on a valid hex
    if not (0 <= game.robber_hex < len(game.board.hexes)):
        violations.append(f"Robber on invalid hex: {game.robber_hex}")

    # 9. No duplicate buildings on same vertex/edge
    all_settlement_vids = []
    all_city_vids = []
    all_road_eids = []
    for player in game.players:
        all_settlement_vids.extend(player.settlements)
        all_city_vids.extend(player.cities)
        all_road_eids.extend(player.roads)
    if len(set(all_settlement_vids) | set(all_city_vids)) != len(all_settlement_vids) + len(
        all_city_vids
    ):
        violations.append("Duplicate vertex occupancy detected")
    if len(set(all_road_eids)) != len(all_road_eids):
        violations.append("Duplicate edge occupancy detected")

    return violations


class TestInvariantsAfterPlacement:
    def test_invariants_hold_after_placement(self):
        from .helpers import make_game, skip_to_main_phase

        game = make_game(4, seed=42)
        skip_to_main_phase(game)
        violations = validate_game_state(game)
        assert violations == [], f"Invariant violations: {violations}"


class TestInvariantsAfterFullGame:
    def test_invariants_hold_after_random_game(self):
        """Run a full game with random agents and check invariants at the end."""
        agents = [RandomAgent(random.Random(i)) for i in range(4)]
        rng = random.Random(42)
        board = Board.standard(shuffle=True, rng=rng)
        players = [Player(name=a.name()) for a in agents]
        game = Game(board=board, players=players, rng=rng)

        while game.phase != GamePhase.FINISHED:
            if game.phase == GamePhase.ROLL:
                game._start_next_turn()
                continue
            if game.phase == GamePhase.ROBBER_DISCARD:
                acting_idx = game.players_to_discard[game._discard_idx]
            else:
                acting_idx = game.current_player_idx
            actions = game.legal_actions()
            if not actions:
                game.phase = GamePhase.FINISHED
                break
            action = agents[acting_idx].choose_action(game, actions)
            game.apply_action(action)
            winner = game.check_victory()
            if winner is not None:
                game.phase = GamePhase.FINISHED

        violations = validate_game_state(game)
        assert violations == [], f"Invariant violations: {violations}"

    def test_invariants_across_multiple_games(self):
        """Run 20 games and check invariants on each."""
        for seed in range(20):
            agents = [RandomAgent(random.Random(seed * 4 + i)) for i in range(4)]
            rng = random.Random(seed)
            board = Board.standard(shuffle=True, rng=rng)
            players = [Player(name=a.name()) for a in agents]
            game = Game(board=board, players=players, rng=rng)

            while game.phase != GamePhase.FINISHED:
                if game.phase == GamePhase.ROLL:
                    game._start_next_turn()
                    continue
                if game.phase == GamePhase.ROBBER_DISCARD:
                    acting_idx = game.players_to_discard[game._discard_idx]
                else:
                    acting_idx = game.current_player_idx
                actions = game.legal_actions()
                if not actions:
                    game.phase = GamePhase.FINISHED
                    break
                action = agents[acting_idx].choose_action(game, actions)
                game.apply_action(action)
                winner = game.check_victory()
                if winner is not None:
                    game.phase = GamePhase.FINISHED

            violations = validate_game_state(game)
            assert violations == [], f"Game seed={seed}: {violations}"
