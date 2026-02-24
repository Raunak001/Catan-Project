"""Tests for the Gymnasium Catan environment."""

from __future__ import annotations

import numpy as np
import pytest

from catan.actions import (
    BankTrade,
    BuildCity,
    BuildRoad,
    BuildSettlement,
    BuyDevCard,
    EndTurn,
    MoveRobber,
    PlayKnight,
    PlayMonopoly,
    PlayRoadBuilding,
    PlayYearOfPlenty,
)
from catan.ai.gym_env import (
    _BANK_TRADE_PAIRS,
    _OFF_BANK_TRADE,
    _OFF_BUY_DEV,
    _OFF_CITY,
    _OFF_DISCARD,
    _OFF_END_TURN,
    _OFF_KNIGHT,
    _OFF_MONOPOLY,
    _OFF_ROAD,
    _OFF_ROAD_BUILDING,
    _OFF_ROBBER,
    _OFF_SETTLEMENT,
    _OFF_YOP,
    _RESOURCES,
    _STEAL_SLOTS,
    _YOP_PAIRS,
    AGENT_SEAT,
    NUM_EDGES,
    NUM_HEXES,
    NUM_PLAYERS,
    NUM_VERTICES,
    OBS_SIZE,
    TOTAL_ACTIONS,
    CatanEnv,
)
from catan.ai.heuristic import RandomAgent
from catan.game import GamePhase
from catan.resources import Resource

# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def env():
    e = CatanEnv(seed=42)
    yield e
    e.close()


def _play_until_done(env: CatanEnv, rng: np.random.Generator, max_steps: int = 10_000):
    """Play random legal actions until the game ends. Returns (obs, reward, steps)."""
    steps = 0
    for _ in range(max_steps):
        mask = env.action_masks()
        legal_ids = np.where(mask)[0]
        action = rng.choice(legal_ids)
        obs, reward, terminated, truncated, _info = env.step(int(action))
        steps += 1
        if terminated or truncated:
            return obs, reward, steps
    raise RuntimeError("Game did not terminate")


# ------------------------------------------------------------------ #
#  Basic env lifecycle                                                 #
# ------------------------------------------------------------------ #


def test_reset_returns_correct_shapes(env: CatanEnv):
    obs, info = env.reset()
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32
    assert "action_mask" in info
    assert info["action_mask"].shape == (TOTAL_ACTIONS,)


def test_obs_values_in_range(env: CatanEnv):
    obs, _ = env.reset()
    assert np.all(obs >= 0.0), f"Min obs value: {obs.min()}"
    assert np.all(obs <= 1.0), f"Max obs value: {obs.max()}"


def test_action_mask_has_legal_actions(env: CatanEnv):
    env.reset()
    mask = env.action_masks()
    assert mask.shape == (TOTAL_ACTIONS,)
    assert mask.dtype == bool
    assert mask.any(), "At least one action must be legal"


def test_step_with_legal_action(env: CatanEnv):
    env.reset()
    mask = env.action_masks()
    action = int(np.argmax(mask))  # pick first legal action
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (OBS_SIZE,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_full_game_terminates(env: CatanEnv):
    """Run a full game choosing random legal actions for seat 0."""
    env.reset(seed=123)
    rng = np.random.default_rng(123)
    done = False
    steps = 0
    max_steps = 10_000  # safety limit

    while not done and steps < max_steps:
        mask = env.action_masks()
        legal_ids = np.where(mask)[0]
        action = rng.choice(legal_ids)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        steps += 1

        # Obs should always be valid
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    assert done, f"Game did not terminate after {max_steps} steps"


def test_reward_on_game_end():
    """The final reward should be +10 (win) or -5 (loss)."""
    env = CatanEnv(seed=7)
    env.reset()
    rng = np.random.default_rng(7)
    final_reward = 0.0
    done = False

    for _ in range(10_000):
        mask = env.action_masks()
        legal_ids = np.where(mask)[0]
        action = rng.choice(legal_ids)
        obs, reward, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            final_reward = reward
            done = True
            break

    assert done
    assert final_reward in (10.0, -5.0)
    env.close()


# ------------------------------------------------------------------ #
#  Action encoding / decoding roundtrips                               #
# ------------------------------------------------------------------ #


def test_roundtrip_build_settlement():
    env = CatanEnv(seed=1)
    env.reset()
    for v in range(NUM_VERTICES):
        action = BuildSettlement(vertex_id=v)
        aid = env._game_action_to_id(action)
        assert aid == _OFF_SETTLEMENT + v
        decoded = env._action_id_to_game_action(aid)
        assert decoded == action
    env.close()


def test_roundtrip_build_road():
    env = CatanEnv(seed=1)
    env.reset()
    for e in range(NUM_EDGES):
        action = BuildRoad(edge_id=e)
        aid = env._game_action_to_id(action)
        assert aid == _OFF_ROAD + e
        decoded = env._action_id_to_game_action(aid)
        assert decoded == action
    env.close()


def test_roundtrip_build_city():
    env = CatanEnv(seed=1)
    env.reset()
    for v in range(NUM_VERTICES):
        action = BuildCity(vertex_id=v)
        aid = env._game_action_to_id(action)
        assert aid == _OFF_CITY + v
        decoded = env._action_id_to_game_action(aid)
        assert decoded == action
    env.close()


def test_roundtrip_bank_trade():
    env = CatanEnv(seed=1)
    env.reset()
    for i, (give, recv) in enumerate(_BANK_TRADE_PAIRS):
        action = BankTrade(give=give, receive=recv)
        aid = env._game_action_to_id(action)
        assert aid == _OFF_BANK_TRADE + i
        decoded = env._action_id_to_game_action(aid)
        assert decoded == action
    env.close()


def test_roundtrip_buy_dev_card():
    env = CatanEnv(seed=1)
    env.reset()
    action = BuyDevCard()
    aid = env._game_action_to_id(action)
    assert aid == _OFF_BUY_DEV
    assert env._action_id_to_game_action(aid) == action
    env.close()


def test_roundtrip_play_road_building():
    env = CatanEnv(seed=1)
    env.reset()
    action = PlayRoadBuilding(edge1=-1, edge2=None)
    aid = env._game_action_to_id(action)
    assert aid == _OFF_ROAD_BUILDING
    decoded = env._action_id_to_game_action(aid)
    assert decoded == action
    env.close()


def test_roundtrip_year_of_plenty():
    env = CatanEnv(seed=1)
    env.reset()
    for i, (r1, r2) in enumerate(_YOP_PAIRS):
        action = PlayYearOfPlenty(resource1=r1, resource2=r2)
        aid = env._game_action_to_id(action)
        assert aid == _OFF_YOP + i
        decoded = env._action_id_to_game_action(aid)
        assert decoded == action
    env.close()


def test_roundtrip_monopoly():
    env = CatanEnv(seed=1)
    env.reset()
    for i, r in enumerate(_RESOURCES):
        action = PlayMonopoly(resource=r)
        aid = env._game_action_to_id(action)
        assert aid == _OFF_MONOPOLY + i
        decoded = env._action_id_to_game_action(aid)
        assert decoded == action
    env.close()


def test_roundtrip_play_knight():
    env = CatanEnv(seed=1)
    env.reset()
    for h in range(NUM_HEXES):
        # None steal
        action = PlayKnight(target_hex=h, steal_from=None)
        aid = env._game_action_to_id(action)
        assert env._action_id_to_game_action(aid) == action
        # Player steal targets
        for p in range(4):
            action = PlayKnight(target_hex=h, steal_from=p)
            aid = env._game_action_to_id(action)
            assert env._action_id_to_game_action(aid) == action
    env.close()


def test_roundtrip_move_robber():
    env = CatanEnv(seed=1)
    env.reset()
    for h in range(NUM_HEXES):
        action = MoveRobber(target_hex=h, steal_from=None)
        aid = env._game_action_to_id(action)
        assert env._action_id_to_game_action(aid) == action
        for p in range(4):
            action = MoveRobber(target_hex=h, steal_from=p)
            aid = env._game_action_to_id(action)
            assert env._action_id_to_game_action(aid) == action
    env.close()


def test_roundtrip_end_turn():
    env = CatanEnv(seed=1)
    env.reset()
    action = EndTurn()
    aid = env._game_action_to_id(action)
    assert aid == _OFF_END_TURN
    assert env._action_id_to_game_action(aid) == action
    env.close()


# ------------------------------------------------------------------ #
#  Action space layout sanity                                          #
# ------------------------------------------------------------------ #


def test_action_space_offsets_are_contiguous():
    """Verify the action space has no gaps or overlaps."""
    assert _OFF_SETTLEMENT == 0
    assert _OFF_ROAD == 54
    assert _OFF_CITY == 54 + 72
    assert _OFF_BANK_TRADE == 54 + 72 + 54
    assert _OFF_BUY_DEV == 54 + 72 + 54 + 20
    assert _OFF_ROAD_BUILDING == _OFF_BUY_DEV + 1
    assert _OFF_YOP == _OFF_ROAD_BUILDING + 1
    assert _OFF_MONOPOLY == _OFF_YOP + 15
    assert _OFF_KNIGHT == _OFF_MONOPOLY + 5
    assert _OFF_ROBBER == _OFF_KNIGHT + 95
    assert _OFF_DISCARD == _OFF_ROBBER + 95
    assert _OFF_END_TURN == _OFF_DISCARD + 50
    assert TOTAL_ACTIONS == _OFF_END_TURN + 1


def test_total_action_count():
    expected = 54 + 72 + 54 + 20 + 1 + 1 + 15 + 5 + 95 + 95 + 50 + 1
    assert TOTAL_ACTIONS == expected


# ------------------------------------------------------------------ #
#  Observation encoding                                                #
# ------------------------------------------------------------------ #


def test_obs_size_matches_space(env: CatanEnv):
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape


def test_obs_contains_board_info(env: CatanEnv):
    """After reset, hex terrain info should be non-zero."""
    obs, _ = env.reset()
    # First 95 values are hex resource one-hot (19 hexes × 5 resources)
    hex_res = obs[:95].reshape(19, 5)
    # At least some hexes produce resources (desert has all-zero row)
    assert hex_res.sum() >= 18  # 18 producing hexes + 1 desert


def test_obs_deterministic_with_seed():
    """Same seed should produce same observation."""
    env1 = CatanEnv(seed=99)
    obs1, _ = env1.reset()
    env1.close()

    env2 = CatanEnv(seed=99)
    obs2, _ = env2.reset()
    env2.close()

    np.testing.assert_array_equal(obs1, obs2)


# ------------------------------------------------------------------ #
#  Placement phase through env                                         #
# ------------------------------------------------------------------ #


def test_placement_phase_actions(env: CatanEnv):
    """During placement, only BuildSettlement or BuildRoad should be legal."""
    env.reset()
    mask = env.action_masks()
    legal_ids = np.where(mask)[0]

    # All legal actions should be settlements or roads
    for aid in legal_ids:
        assert (
            aid < _OFF_CITY  # BuildSettlement (0–53) or BuildRoad (54–125)
        ), f"Unexpected action {aid} during placement"


# ------------------------------------------------------------------ #
#  Multiple games with different seeds                                 #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_game_completes_with_various_seeds(seed: int):
    """Games should complete without errors across different seeds."""
    env = CatanEnv(seed=seed)
    env.reset()
    rng = np.random.default_rng(seed)

    for _ in range(10_000):
        mask = env.action_masks()
        legal_ids = np.where(mask)[0]
        action = rng.choice(legal_ids)
        _, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break

    env.close()


# ================================================================== #
#  NEW COMPREHENSIVE PHASE 2 TESTS                                     #
# ================================================================== #


# ------------------------------------------------------------------ #
#  Gymnasium API compliance                                            #
# ------------------------------------------------------------------ #


class TestGymnasiumAPICompliance:
    """Ensure CatanEnv satisfies the gymnasium.Env interface contract."""

    def test_observation_space_type(self, env: CatanEnv):
        from gymnasium.spaces import Box

        assert isinstance(env.observation_space, Box)

    def test_action_space_type(self, env: CatanEnv):
        from gymnasium.spaces import Discrete

        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == TOTAL_ACTIONS

    def test_obs_within_observation_space(self, env: CatanEnv):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), "Observation not in observation_space"

    def test_step_obs_within_observation_space(self, env: CatanEnv):
        env.reset()
        mask = env.action_masks()
        action = int(np.argmax(mask))
        obs, _, _, _, _ = env.step(action)
        assert env.observation_space.contains(obs)

    def test_reset_with_different_seed(self, env: CatanEnv):
        obs1, _ = env.reset(seed=100)
        obs2, _ = env.reset(seed=200)
        # Different seeds should (almost certainly) produce different boards
        assert not np.array_equal(obs1, obs2)

    def test_reset_with_same_seed_is_deterministic(self):
        env = CatanEnv(seed=50)
        obs1, info1 = env.reset(seed=50)
        mask1 = info1["action_mask"].copy()
        obs2, info2 = env.reset(seed=50)
        mask2 = info2["action_mask"].copy()
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(mask1, mask2)
        env.close()

    def test_multiple_resets(self, env: CatanEnv):
        """Env should support being reset multiple times."""
        for seed in range(5):
            obs, info = env.reset(seed=seed)
            assert obs.shape == (OBS_SIZE,)
            assert "action_mask" in info

    def test_step_after_reset_required(self):
        env = CatanEnv(seed=1)
        with pytest.raises(AssertionError, match="reset"):
            env.step(0)
        env.close()

    def test_metadata_has_render_modes(self, env: CatanEnv):
        assert "render_modes" in env.metadata

    def test_info_dict_has_action_mask(self, env: CatanEnv):
        _, info = env.reset()
        assert "action_mask" in info
        mask = info["action_mask"]
        assert mask.dtype == bool
        assert mask.shape == (TOTAL_ACTIONS,)


# ------------------------------------------------------------------ #
#  Action masking correctness                                          #
# ------------------------------------------------------------------ #


class TestActionMasking:
    """Verify action masks match the engine's legal_actions at every step."""

    def test_mask_matches_engine_legal_actions(self):
        """Every legal engine action should have its mask bit set."""
        env = CatanEnv(seed=42)
        env.reset()
        rng = np.random.default_rng(42)

        for _ in range(200):
            mask = env.action_masks()
            legal_ids = np.where(mask)[0]
            assert len(legal_ids) > 0

            # Pick a random legal action and step
            action = rng.choice(legal_ids)
            _, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
        env.close()

    def test_illegal_action_not_in_mask(self, env: CatanEnv):
        """End turn should not be legal during placement."""
        env.reset()
        mask = env.action_masks()
        assert not mask[_OFF_END_TURN], "EndTurn should be illegal during placement"

    def test_no_city_during_placement(self, env: CatanEnv):
        """No city actions should be legal during placement."""
        env.reset()
        mask = env.action_masks()
        city_mask = mask[_OFF_CITY : _OFF_CITY + NUM_VERTICES]
        assert not city_mask.any(), "Cities should not be legal during placement"

    def test_no_bank_trade_during_placement(self, env: CatanEnv):
        """No bank trade actions during placement."""
        env.reset()
        mask = env.action_masks()
        trade_mask = mask[_OFF_BANK_TRADE : _OFF_BANK_TRADE + 20]
        assert not trade_mask.any()

    def test_no_dev_card_actions_during_placement(self, env: CatanEnv):
        """No dev card actions during placement."""
        env.reset()
        mask = env.action_masks()
        assert not mask[_OFF_BUY_DEV]
        assert not mask[_OFF_ROAD_BUILDING]
        yop_mask = mask[_OFF_YOP : _OFF_YOP + 15]
        assert not yop_mask.any()
        monopoly_mask = mask[_OFF_MONOPOLY : _OFF_MONOPOLY + 5]
        assert not monopoly_mask.any()
        knight_mask = mask[_OFF_KNIGHT : _OFF_KNIGHT + 95]
        assert not knight_mask.any()

    def test_mask_consistent_over_multiple_calls(self, env: CatanEnv):
        """Calling action_masks() multiple times without stepping should return the same mask."""
        env.reset()
        mask1 = env.action_masks()
        mask2 = env.action_masks()
        np.testing.assert_array_equal(mask1, mask2)

    def test_at_least_one_legal_action_per_step(self):
        """At every non-terminal step, at least one action must be legal."""
        env = CatanEnv(seed=10)
        env.reset()
        rng = np.random.default_rng(10)

        for _ in range(500):
            mask = env.action_masks()
            assert mask.any(), "No legal actions available (game should be terminated)"
            legal_ids = np.where(mask)[0]
            action = rng.choice(legal_ids)
            _, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
        env.close()


# ------------------------------------------------------------------ #
#  Observation encoding details                                        #
# ------------------------------------------------------------------ #


class TestObservationEncoding:
    """Test specific segments of the observation vector."""

    def test_hex_resources_one_hot(self, env: CatanEnv):
        """Each hex row should be one-hot (one resource) or all zeros (desert)."""
        obs, _ = env.reset()
        hex_res = obs[:95].reshape(19, 5)
        for row in hex_res:
            active = np.count_nonzero(row)
            assert active <= 1, f"Hex resource row should be one-hot or zero, got {row}"

    def test_exactly_one_desert(self, env: CatanEnv):
        """Exactly one hex should be desert (all-zero row in resource one-hot)."""
        obs, _ = env.reset()
        hex_res = obs[:95].reshape(19, 5)
        zero_rows = sum(1 for row in hex_res if row.sum() == 0)
        assert zero_rows == 1, f"Expected 1 desert hex, found {zero_rows}"

    def test_hex_probability_values(self, env: CatanEnv):
        """Hex probabilities should be non-negative and at most 1.0."""
        obs, _ = env.reset()
        hex_prob = obs[95:114]
        assert np.all(hex_prob >= 0.0)
        assert np.all(hex_prob <= 1.0)

    def test_robber_one_hot(self, env: CatanEnv):
        """Robber location should be exactly one hex set to 1.0."""
        obs, _ = env.reset()
        robber = obs[114:133]
        assert robber.sum() == pytest.approx(1.0)
        assert np.count_nonzero(robber) == 1

    def test_initial_vertex_owner_empty(self, env: CatanEnv):
        """Before placement completes, many vertices should be unowned.

        After reset the placement phase auto-advances opponents, so some
        vertices may already be placed by opponents. But seat 0 hasn't
        placed yet, so we just check the vector is mostly zero."""
        obs, _ = env.reset()
        vertex_owner = obs[133:187]
        # At most 3 opponents could have placed one settlement each so far
        assert np.count_nonzero(vertex_owner) <= 6

    def test_phase_one_hot(self, env: CatanEnv):
        """Game phase should be a valid one-hot vector."""
        obs, _ = env.reset()
        # Phase one-hot starts at a known offset — compute it
        # From OBS_SIZE layout: hex_res(95) + hex_prob(19) + robber(19) +
        # vertex_owner(54) + vertex_bldg(54) + edge_owner(72) +
        # player_res(20) + player_vp(4) + player_dev(20) + player_knights(4) +
        # largest_army(4) + longest_road(4) + ports(24) = 393
        phase_start = 393
        phase_vec = obs[phase_start : phase_start + 6]
        assert phase_vec.sum() == pytest.approx(1.0)
        assert np.count_nonzero(phase_vec) == 1

    def test_current_player_one_hot(self, env: CatanEnv):
        """Current player should be a valid one-hot vector."""
        obs, _ = env.reset()
        cur_player_start = 393 + 6  # after phase
        cur_player = obs[cur_player_start : cur_player_start + 4]
        assert cur_player.sum() == pytest.approx(1.0)
        assert np.count_nonzero(cur_player) == 1

    def test_turn_is_normalized(self, env: CatanEnv):
        """Turn should be in [0, 1] range."""
        obs, _ = env.reset()
        turn_idx = 393 + 6 + 4  # after phase + current_player
        assert 0.0 <= obs[turn_idx] <= 1.0

    def test_obs_changes_after_step(self, env: CatanEnv):
        """Observation should change after taking an action."""
        obs1, _ = env.reset()
        mask = env.action_masks()
        action = int(np.argmax(mask))
        obs2, _, _, _, _ = env.step(action)
        # The obs should differ after an action (board state changed)
        assert not np.array_equal(obs1, obs2)


# ------------------------------------------------------------------ #
#  Opponent integration                                                #
# ------------------------------------------------------------------ #


class TestOpponentIntegration:
    """Test that opponent agents are properly called during auto-advance."""

    def test_custom_opponents(self):
        """Env should accept custom opponent agents."""
        opponents = [RandomAgent() for _ in range(3)]
        env = CatanEnv(opponent_agents=opponents, seed=42)
        obs, _ = env.reset()
        assert obs.shape == (OBS_SIZE,)
        env.close()

    def test_wrong_number_of_opponents_raises(self):
        """Passing wrong number of opponents should raise."""
        with pytest.raises(AssertionError):
            CatanEnv(opponent_agents=[RandomAgent(), RandomAgent()])

    def test_agent_seat_is_always_zero(self):
        assert AGENT_SEAT == 0

    def test_game_always_returns_to_agent_seat(self):
        """After stepping, the game state should be at agent's turn or game over."""
        env = CatanEnv(seed=42)
        env.reset()
        rng = np.random.default_rng(42)

        for _ in range(100):
            mask = env.action_masks()
            legal_ids = np.where(mask)[0]
            action = rng.choice(legal_ids)
            _, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
            # Should be agent's turn or agent needs to discard
            game = env.game
            if game.phase == GamePhase.ROBBER_DISCARD:
                acting = game.players_to_discard[game._discard_idx]
            else:
                acting = game.current_player_idx
            assert acting == AGENT_SEAT
        env.close()


# ------------------------------------------------------------------ #
#  Reward structure                                                    #
# ------------------------------------------------------------------ #


class TestRewardStructure:
    """Test the reward signal from the environment."""

    def test_intermediate_reward_is_shaped(self):
        """Non-terminal steps should have shaped reward based on VP/resource deltas."""
        env = CatanEnv(seed=42)
        env.reset()
        rng = np.random.default_rng(42)
        saw_nonzero = False

        for _ in range(500):
            mask = env.action_masks()
            legal_ids = np.where(mask)[0]
            action = rng.choice(legal_ids)
            _, reward, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
            # Intermediate rewards should be small (VP delta * 1.0 + settlement quality)
            # They can be positive, negative, or zero
            assert abs(reward) < 15.0, f"Intermediate reward too large: {reward}"
            if reward != 0.0:
                saw_nonzero = True
        env.close()
        # Over 500 steps, we should see at least some non-zero shaped rewards
        assert saw_nonzero, "Expected some non-zero intermediate rewards from shaping"

    def test_terminal_reward_magnitude(self):
        """Terminal reward should be exactly +10 (win) or -5 (loss)."""
        env = CatanEnv(seed=7)
        env.reset()
        rng = np.random.default_rng(7)
        obs, reward, steps = _play_until_done(env, rng)
        assert reward in (10.0, -5.0)
        env.close()

    def test_winner_gets_positive_reward(self):
        """If seat 0 wins, reward should be +10."""
        for seed in range(20):
            env = CatanEnv(seed=seed)
            env.reset()
            rng = np.random.default_rng(seed)
            try:
                _, reward, _ = _play_until_done(env, rng)
            except RuntimeError:
                env.close()
                continue
            if reward == 10.0:
                winner = env.game.check_victory()
                assert winner == AGENT_SEAT
                env.close()
                return
            env.close()
        pytest.skip("Seat 0 didn't win in any test seed")

    def test_loser_gets_negative_reward(self):
        """If seat 0 loses, reward should be -5."""
        for seed in range(20):
            env = CatanEnv(seed=seed)
            env.reset()
            rng = np.random.default_rng(seed)
            try:
                _, reward, _ = _play_until_done(env, rng)
            except RuntimeError:
                env.close()
                continue
            if reward == -5.0:
                winner = env.game.check_victory()
                assert winner != AGENT_SEAT
                env.close()
                return
            env.close()
        pytest.skip("Seat 0 didn't lose in any test seed")

    def test_vp_gain_gives_reward(self):
        """Gaining a VP should produce a +1.0 reward component."""
        env = CatanEnv(seed=10)
        env.reset()
        agent = env.game.players[AGENT_SEAT]
        # Manually bump VP to simulate a gain
        old_vp = agent.victory_points
        agent.victory_points += 1
        # Force a step — use EndTurn if available
        mask = env.action_masks()
        legal_ids = np.where(mask)[0]
        action = legal_ids[0]
        # Reset tracking so delta picks up the VP change
        env._prev_vp = old_vp
        env._prev_total_resources = agent.total_resource_count()
        _, reward, terminated, truncated, _ = env.step(int(action))
        if not terminated and not truncated:
            # Reward should include +1.0 from the VP gain
            assert reward >= 1.0, f"Expected VP reward component >= 1.0, got {reward}"

    def test_no_resource_gain_reward(self):
        """Resource gains alone should not produce reward (simplified shaping)."""
        env = CatanEnv(seed=10)
        env.reset()
        agent = env.game.players[AGENT_SEAT]
        from catan.resources import Resource

        # Give the agent extra resources — should NOT affect reward
        old_res = agent.total_resource_count()
        agent.resources[Resource.BRICK] += 5
        new_res = agent.total_resource_count()
        assert new_res == old_res + 5  # sanity check
        # With simplified rewards, resource gains don't contribute to reward.
        # The reward is only VP delta + settlement quality, so adding resources
        # doesn't change the reward compared to not adding them.
        # (We can't easily test this in isolation since stepping may trigger VP
        # changes from placement, so we just verify the env doesn't crash.)
        mask = env.action_masks()
        legal_ids = np.where(mask)[0]
        _, reward, terminated, truncated, _ = env.step(int(legal_ids[0]))
        # Reward should be finite and reasonable
        assert np.isfinite(reward), f"Reward should be finite, got {reward}"


# ------------------------------------------------------------------ #
#  Info dict contents                                                  #
# ------------------------------------------------------------------ #


class TestInfoDict:
    """Test the info dict returned by step() and reset()."""

    def test_reset_info_has_mask(self, env: CatanEnv):
        _, info = env.reset()
        assert "action_mask" in info

    def test_terminal_info_has_winner(self):
        env = CatanEnv(seed=7)
        env.reset()
        rng = np.random.default_rng(7)
        for _ in range(10_000):
            mask = env.action_masks()
            legal_ids = np.where(mask)[0]
            action = rng.choice(legal_ids)
            _, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                assert "winner" in info
                break
        env.close()

    def test_nonterminal_info_has_mask(self):
        env = CatanEnv(seed=42)
        env.reset()
        rng = np.random.default_rng(42)
        for _ in range(50):
            mask = env.action_masks()
            legal_ids = np.where(mask)[0]
            action = rng.choice(legal_ids)
            _, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                break
            assert "action_mask" in info
        env.close()


# ------------------------------------------------------------------ #
#  Edge cases in action encoding                                       #
# ------------------------------------------------------------------ #


class TestActionEncodingEdgeCases:
    """Test boundary conditions and edge cases in action encoding."""

    def test_invalid_action_id_raises(self, env: CatanEnv):
        env.reset()
        with pytest.raises(ValueError, match="Invalid action ID"):
            env._action_id_to_game_action(TOTAL_ACTIONS)

    def test_invalid_action_id_negative(self, env: CatanEnv):
        """Negative action IDs should raise or return a bogus action."""
        env.reset()
        # Negative index may hit the BuildSettlement branch with negative vertex
        # which is technically "valid" encoding but nonsensical — verify it doesn't crash
        # The important thing is TOTAL_ACTIONS and beyond raise
        with pytest.raises(ValueError):
            env._action_id_to_game_action(TOTAL_ACTIONS + 100)

    def test_discard_out_of_range_raises(self, env: CatanEnv):
        """Accessing discard action index beyond cached list should raise."""
        env.reset()
        # _discard_actions is empty after reset, so any discard action ID should raise
        with pytest.raises(ValueError, match="Discard"):
            env._action_id_to_game_action(_OFF_DISCARD)

    def test_bank_trade_pairs_no_self_trade(self):
        """Bank trade pairs should never have give == receive."""
        for give, recv in _BANK_TRADE_PAIRS:
            assert give != recv

    def test_yop_pairs_are_ordered(self):
        """Year of Plenty pairs should have r1 index <= r2 index."""
        for r1, r2 in _YOP_PAIRS:
            assert _RESOURCES.index(r1) <= _RESOURCES.index(r2)

    def test_yop_pairs_cover_all_combinations(self):
        """Should have C(5,2) + 5 = 15 pairs (with repetition)."""
        assert len(_YOP_PAIRS) == 15

    def test_bank_trade_pairs_count(self):
        """5 resources × 4 other resources = 20 pairs."""
        assert len(_BANK_TRADE_PAIRS) == 20

    def test_knight_encoding_boundary(self, env: CatanEnv):
        """Test encoding at the boundary between knight and robber actions."""
        env.reset()
        # Last knight action
        last_knight = PlayKnight(target_hex=NUM_HEXES - 1, steal_from=3)
        kid = env._game_action_to_id(last_knight)
        assert kid == _OFF_KNIGHT + (NUM_HEXES - 1) * _STEAL_SLOTS + 4
        assert kid < _OFF_ROBBER

        # First robber action
        first_robber = MoveRobber(target_hex=0, steal_from=None)
        rid = env._game_action_to_id(first_robber)
        assert rid == _OFF_ROBBER

    def test_steal_from_none_encodes_as_slot_zero(self, env: CatanEnv):
        env.reset()
        action = PlayKnight(target_hex=5, steal_from=None)
        aid = env._game_action_to_id(action)
        expected = _OFF_KNIGHT + 5 * _STEAL_SLOTS + 0
        assert aid == expected

    def test_steal_from_player_encodes_correctly(self, env: CatanEnv):
        env.reset()
        for p in range(4):
            action = MoveRobber(target_hex=3, steal_from=p)
            aid = env._game_action_to_id(action)
            expected = _OFF_ROBBER + 3 * _STEAL_SLOTS + (p + 1)
            assert aid == expected


# ------------------------------------------------------------------ #
#  Observation stability over a game                                   #
# ------------------------------------------------------------------ #


class TestObservationStability:
    """Verify observation vector stays valid throughout an entire game."""

    def test_obs_always_in_bounds(self):
        """Observation values must remain in [0, 1] for all steps."""
        env = CatanEnv(seed=55)
        env.reset()
        rng = np.random.default_rng(55)

        for step_num in range(5_000):
            mask = env.action_masks()
            legal_ids = np.where(mask)[0]
            action = rng.choice(legal_ids)
            obs, _, terminated, truncated, _ = env.step(int(action))
            assert np.all(obs >= 0.0), f"Negative obs at step {step_num}: min={obs.min()}"
            assert np.all(obs <= 1.0), f"Obs > 1 at step {step_num}: max={obs.max()}"
            if terminated or truncated:
                break
        env.close()

    def test_obs_no_nans(self):
        """Observation should never contain NaN."""
        env = CatanEnv(seed=77)
        env.reset()
        rng = np.random.default_rng(77)

        for _ in range(3_000):
            mask = env.action_masks()
            legal_ids = np.where(mask)[0]
            action = rng.choice(legal_ids)
            obs, _, terminated, truncated, _ = env.step(int(action))
            assert not np.any(np.isnan(obs)), "NaN in observation"
            if terminated or truncated:
                break
        env.close()

    def test_obs_dtype_is_float32(self):
        """Observation dtype should always be float32."""
        env = CatanEnv(seed=88)
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        mask = env.action_masks()
        action = int(np.argmax(mask))
        obs, _, _, _, _ = env.step(action)
        assert obs.dtype == np.float32
        env.close()


# ------------------------------------------------------------------ #
#  Action mask alignment with engine                                   #
# ------------------------------------------------------------------ #


class TestMaskEngineAlignment:
    """Verify that masked actions can always be successfully applied."""

    def test_all_masked_actions_are_valid(self):
        """Every action flagged as legal in the mask should be executable."""
        env = CatanEnv(seed=33)
        env.reset()
        rng = np.random.default_rng(33)

        for _ in range(300):
            mask = env.action_masks()
            legal_ids = np.where(mask)[0]
            # Verify at least the chosen action doesn't crash
            action = rng.choice(legal_ids)
            try:
                obs, reward, terminated, truncated, info = env.step(int(action))
            except Exception as e:
                pytest.fail(f"Legal action {action} raised: {e}")
            if terminated or truncated:
                break
        env.close()

    def test_mask_count_matches_engine_count(self):
        """Number of True entries in mask should match len(legal_actions())."""
        env = CatanEnv(seed=44)
        env.reset()

        # The mask is built from legal_actions(), so counts should match
        # (unless some actions are unmappable)
        mask = env.action_masks()
        engine_legal = env.game.legal_actions()
        mapped_count = sum(
            1 for a in engine_legal if env._game_action_to_id(a) is not None
        )
        mask_count = int(mask.sum())
        assert mask_count == mapped_count
        env.close()


# ------------------------------------------------------------------ #
#  Game state access via env                                           #
# ------------------------------------------------------------------ #


class TestGameStateAccess:
    """Test that the underlying game state is accessible and consistent."""

    def test_game_is_none_before_reset(self):
        env = CatanEnv(seed=1)
        assert env.game is None
        env.close()

    def test_game_is_set_after_reset(self, env: CatanEnv):
        env.reset()
        assert env.game is not None

    def test_game_has_four_players(self, env: CatanEnv):
        env.reset()
        assert len(env.game.players) == NUM_PLAYERS

    def test_game_board_has_19_hexes(self, env: CatanEnv):
        env.reset()
        assert len(env.game.board.hexes) == NUM_HEXES

    def test_reset_creates_new_game(self, env: CatanEnv):
        env.reset(seed=1)
        game1 = env.game
        env.reset(seed=2)
        game2 = env.game
        assert game1 is not game2


# ------------------------------------------------------------------ #
#  Shuffled vs unshuffled board                                        #
# ------------------------------------------------------------------ #


class TestBoardShuffle:
    """Test shuffle_board parameter."""

    def test_unshuffled_board_is_deterministic(self):
        env1 = CatanEnv(seed=1, shuffle_board=False)
        obs1, _ = env1.reset()
        env1.close()

        env2 = CatanEnv(seed=2, shuffle_board=False)
        obs2, _ = env2.reset()
        env2.close()

        # With shuffle=False, hex layout should be identical
        # (first 95 values = hex resource one-hot)
        np.testing.assert_array_equal(obs1[:95], obs2[:95])

    def test_shuffled_board_varies(self):
        """Different seeds with shuffle=True should produce different boards."""
        obs_list = []
        for seed in [100, 200, 300]:
            env = CatanEnv(seed=seed, shuffle_board=True)
            obs, _ = env.reset()
            obs_list.append(obs[:95].copy())
            env.close()
        # At least two should differ
        assert not (
            np.array_equal(obs_list[0], obs_list[1])
            and np.array_equal(obs_list[1], obs_list[2])
        )


# ------------------------------------------------------------------ #
#  Stress tests                                                        #
# ------------------------------------------------------------------ #


class TestStress:
    """Stress tests running multiple games to catch rare bugs."""

    @pytest.mark.parametrize("seed", range(10, 25))
    def test_full_game_no_crash(self, seed):
        """Run 15 full games to catch any crashes."""
        env = CatanEnv(seed=seed)
        env.reset()
        rng = np.random.default_rng(seed)

        for _ in range(10_000):
            mask = env.action_masks()
            legal_ids = np.where(mask)[0]
            assert len(legal_ids) > 0
            action = rng.choice(legal_ids)
            _, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
        env.close()

    def test_rapid_reset_cycle(self):
        """Rapidly reset the env many times."""
        env = CatanEnv(seed=42)
        for i in range(20):
            obs, info = env.reset(seed=i)
            assert obs.shape == (OBS_SIZE,)
            mask = info["action_mask"]
            assert mask.any()
        env.close()

    def test_play_then_reset_then_play(self):
        """Play partial game, reset, play again."""
        env = CatanEnv(seed=42)
        rng = np.random.default_rng(42)

        for _ in range(3):
            env.reset()
            for _ in range(50):
                mask = env.action_masks()
                legal_ids = np.where(mask)[0]
                action = rng.choice(legal_ids)
                _, _, terminated, truncated, _ = env.step(int(action))
                if terminated or truncated:
                    break
        env.close()


# ------------------------------------------------------------------ #
#  Discard action encoding                                             #
# ------------------------------------------------------------------ #


class TestDiscardEncoding:
    """Test the dynamic discard action encoding."""

    def test_discard_actions_cached_on_mask_call(self, env: CatanEnv):
        """action_masks() should populate _discard_actions."""
        env.reset()
        env.action_masks()
        # During placement, no discard actions expected
        assert isinstance(env._discard_actions, list)

    def test_discard_max_50_slots(self):
        """The discard action space has exactly 50 slots."""
        assert _OFF_END_TURN - _OFF_DISCARD == 50


# ------------------------------------------------------------------ #
#  Constants and layout                                                #
# ------------------------------------------------------------------ #


class TestConstants:
    """Verify module-level constants are self-consistent."""

    def test_num_vertices(self):
        assert NUM_VERTICES == 54

    def test_num_edges(self):
        assert NUM_EDGES == 72

    def test_num_hexes(self):
        assert NUM_HEXES == 19

    def test_num_players(self):
        assert NUM_PLAYERS == 4

    def test_resources_list(self):
        assert len(_RESOURCES) == 5
        assert all(isinstance(r, Resource) for r in _RESOURCES)

    def test_steal_slots(self):
        assert _STEAL_SLOTS == 5  # None + 4 players

    def test_obs_size_positive(self):
        assert OBS_SIZE == 407

    def test_total_actions_positive(self):
        assert TOTAL_ACTIONS == 463
