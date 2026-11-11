"""EN-02: minimal PCSEnv regressions with synthetic, half-hourly inputs.

All quantities are PCSEnv internal units, not kW/kWh. Strict xfails expose
known discrepancies without changing the simulator or blessing new semantics.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from energy_net.gym_envs.pcs_env import PCSEnv
from energy_net.grid_entities.consumption.consumption_dynamics import CSV_DataConsumptionDynamics
from energy_net.grid_entities.management.price_curve import PriceCurveStrategy


class FixedPriceStrategy(PriceCurveStrategy):
    def calculate_price(self, observation=None):
        return np.full(48, 0.1)


class HalfHourlyPriceStrategy(PriceCurveStrategy):
    def calculate_price(self, observation=None):
        return np.arange(48, dtype=float) / 100


@pytest.fixture
def make_env(tmp_path):
    """Write isolated fixtures; never interpolate or modify research CSVs."""
    environments = []

    def create(loads=None, price_strategy=None):
        if loads is None:
            loads = np.full(96, 3.0)
        timestamps = pd.date_range('2025-07-09', periods=len(loads), freq='30min')
        data_file = tmp_path / 'consumption.csv'
        predictions_file = tmp_path / 'predictions.csv'
        pd.DataFrame({'Datetime': timestamps, 'Consumption': loads}).to_csv(data_file, index=False)
        pd.DataFrame({
            'timestamp': timestamps,
            'predicted_consumption': np.full(len(loads), 3.0),
        }).to_csv(predictions_file, index=False)
        env = PCSEnv(
            test_data_file=str(data_file),
            predictions_file=str(predictions_file),
            price_strategy=price_strategy or FixedPriceStrategy(),
            verbosity=0,
        )
        environments.append(env)
        return env

    yield create
    for env in environments:
        env.close()


def step_from_storage(env, storage, action):
    env.reset(start_date=env.test_start_date, seed=42)
    env.pcs.storage_units[0].reset(initial_level=storage)
    return env.step(np.array([action], dtype=np.float32))


SHORTAGE_DISCREPANCY = pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason='EN-02: post-consumption flag disagrees with battery-uncovered demand; semantics pending EN-03',
)


@pytest.mark.parametrize('storage,load,expected_shortage', [
    (0.0, 0.0, False),
    (2.0, 3.0, True),
    pytest.param(3.0, 3.0, False, marks=SHORTAGE_DISCREPANCY),
    pytest.param(5.0, 3.19168, False, marks=SHORTAGE_DISCREPANCY),
    (6.0, 3.0, False),
    (10.0, 3.0, False),
])
def test_shortage_matches_uncovered_demand(make_env, storage, load, expected_shortage):
    """Conditional contract: shortage means demand the battery could not supply."""
    env = make_env(np.full(96, load))
    _, reward, _, _, info = step_from_storage(env, storage, 0.0)
    assert info['storage_after_units'] == pytest.approx(max(0, storage - load))
    assert info['consumption_units'] == pytest.approx(load)
    assert info['step_money'] == pytest.approx(0)
    assert info['shortage'] == expected_shortage
    assert reward == pytest.approx(-float(expected_shortage))


@pytest.mark.parametrize('storage,action,load,transaction,final_storage', [
    (20.0, 10.0, 5.0, 10.0, 25.0),
    (98.0, 10.0, 3.0, 2.0, 97.0),
    (2.0, -10.0, 0.0, -2.0, 0.0),
    (100.0, 10.0, 0.0, 0.0, 100.0),
    (0.0, -10.0, 0.0, 0.0, 0.0),
])
def test_realized_transaction_and_two_stage_update(make_env, storage, action, load, transaction, final_storage):
    env = make_env(np.full(96, load))
    obs, reward, _, _, info = step_from_storage(env, storage, action)
    assert info['battery_action'] == pytest.approx(action)
    assert info['energy_sold_or_bought_units'] == pytest.approx(transaction)
    assert info['storage_after_units'] == pytest.approx(final_storage)
    assert info['step_money'] == pytest.approx(-0.1 * transaction)
    assert reward == pytest.approx(info['step_money'] - env.shortage_penalty * info['shortage'])
    assert obs[0] == pytest.approx(final_storage / 100)


def test_legacy_residual_demand_accounting(make_env):
    """Preserve −1.3 pending the advisor decision; do not impose grid settlement."""
    env = make_env(np.full(96, 5.0))
    _, reward, _, _, info = step_from_storage(env, 0.0, 3.0)
    assert info['energy_sold_or_bought_units'] == pytest.approx(3)
    assert info['storage_after_units'] == pytest.approx(0)
    assert info['consumption_units'] == pytest.approx(5)
    assert info['step_money'] == pytest.approx(-0.3)
    assert info['shortage']
    assert reward == pytest.approx(-1.3)
    assert env.total_money_earned == pytest.approx(-0.3)
    assert env.total_shortage_penalty == pytest.approx(1)


@pytest.mark.parametrize('phase', ['reset', 'step'])
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='EN-04: PCSEnv uses episode-relative consumption time, ignoring selected day')
def test_selected_date_controls_consumption(make_env, phase):
    loads = np.concatenate([np.full(48, 2.0), np.full(48, 5.0)])
    env = make_env(loads)
    for day, expected in [(0, 2.0), (1, 5.0)]:
        start_date = env.test_start_date + timedelta(days=day)
        obs, _ = env.reset(start_date=start_date)
        assert env.current_datetime == start_date
        if phase == 'reset':
            assert obs[1] * 10 == pytest.approx(expected)
        else:
            _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
            assert info['consumption_units'] == pytest.approx(expected)


def test_csv_dynamics_can_select_two_days(make_env):
    """Localize the date defect to environment wiring, not CSV day indexing."""
    env = make_env(np.concatenate([np.full(48, 2.0), np.full(48, 5.0)]))
    dynamics = env.pcs.consumption_units[0].dynamics
    assert isinstance(dynamics, CSV_DataConsumptionDynamics)
    assert dynamics.get_value(time=0.0) == pytest.approx(2)
    assert dynamics.get_value(time=1.0) == pytest.approx(5)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='EN-04: NaN at an existing CSV timestamp leaks into the observation')
def test_missing_demand_is_rejected_or_kept_out_of_observations(make_env):
    """Allow explicit rejection or finite handling; no interpolation rule chosen."""
    loads = np.full(96, 3.0)
    loads[22] = np.nan
    try:
        env = make_env(loads)
        obs, _ = env.reset(start_date=env.test_start_date)
        assert np.isfinite(obs).all()
        for _ in range(23):
            obs, reward, _, _, info = env.step(np.array([0.0], dtype=np.float32))
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            assert np.isfinite(info['consumption_units'])
    except ValueError as error:
        # An unrelated ValueError must not count as a valid data rejection.
        assert any(word in str(error).lower() for word in ('missing', 'nan', 'finite', 'consumption'))


def test_reset_clears_physical_and_financial_episode_state(make_env):
    env = make_env()
    step_from_storage(env, 0.0, 3.0)
    assert env.current_step == 1
    assert env.total_reward != 0
    assert env.shortage_count == 1
    env.pcs.storage_units[0].reset(initial_level=25)
    obs, info = env.reset(start_date=env.test_start_date, seed=42)
    assert env.current_datetime == env.test_start_date
    assert env.current_step == 0
    assert env.pcs.get_total_storage() == 0
    assert env.last_action is None
    assert info == {}
    for value in (env.total_reward, env.total_money_earned, env.total_shortage_penalty, env.shortage_count):
        assert value == 0
    assert obs[0] == 0
    assert obs[1] == pytest.approx(0.3)


def test_half_hourly_episode_consumption_prices_and_termination(make_env):
    loads = np.arange(96, dtype=float) / 100 + 1
    env = make_env(loads, HalfHourlyPriceStrategy())
    obs, _ = env.reset(start_date=env.test_start_date)
    assert obs[1] * 10 == pytest.approx(loads[0])
    assert env.max_steps == 48
    rewards = []
    for step in range(48):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
        rewards.append(reward)
        assert info['consumption_units'] == pytest.approx(loads[step])
        assert info['current_price'] == pytest.approx(step / 100)
        assert env.current_datetime == env.test_start_date + timedelta(minutes=30 * (step + 1))
        assert bool(terminated) == (step == 47)
        assert not truncated
        assert np.isfinite(obs).all()
    assert env.total_reward == pytest.approx(sum(rewards))
    assert env.current_step == 48
