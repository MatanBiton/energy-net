"""EN-04 demand selection, finite inputs, and partition-local preprocessing."""

import hashlib
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from energy_net.common.demand_data import load_demand_data
from energy_net.grid_entities.consumption.consumption_dynamics import CSV_DataConsumptionDynamics
from energy_net.grid_entities.management.price_curve import PriceCurveStrategy
from energy_net.gym_envs.pcs_env import PCSEnv


class HalfHourlyPriceStrategy(PriceCurveStrategy):
    def calculate_price(self, observation=None):
        return np.arange(48, dtype=float) / 100


@pytest.fixture
def data_files(tmp_path):
    timestamps = pd.date_range('2025-07-09', periods=144, freq='30min')
    data_file = tmp_path / 'demand.csv'
    predictions_file = tmp_path / 'predictions.csv'
    pd.DataFrame({
        'Datetime': timestamps,
        'Consumption': np.arange(144) / 100 + 1,
    }).to_csv(data_file, index=False)
    pd.DataFrame({
        'timestamp': timestamps,
        'predicted_consumption': np.arange(144) / 100 + 2,
        'feature': np.arange(144, dtype=float),
    }).to_csv(predictions_file, index=False)
    return data_file, predictions_file


def create_env(data_files, **kwargs):
    data_file, predictions_file = data_files
    return PCSEnv(test_data_file=data_file, predictions_file=predictions_file,
                  price_strategy=HalfHourlyPriceStrategy(), verbosity=0,
                  data_mode='corrected', **kwargs)


def set_load(data_file, index, value):
    df = pd.read_csv(data_file)
    df.loc[index, 'Consumption'] = value
    df.to_csv(data_file, index=False)


def test_all_selected_samples_and_predictions(data_files):
    env = create_env(data_files)
    for day in (2, 0, 1):
        date = env.test_start_date + timedelta(days=day)
        obs, _ = env.reset(start_date=date)
        assert obs[1] * 10 == pytest.approx(1 + day * 0.48)
        assert obs[3] * 10 == pytest.approx(2 + day * 0.48)
        assert obs[-1] == day * 48
        for step in range(48):
            index = day * 48 + step
            obs, reward, terminated, truncated, info = env.step(np.zeros(1))
            assert info['consumption_units'] == pytest.approx(1 + index / 100)
            assert info['current_price'] == pytest.approx(step / 100)
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert terminated == (step == 47)
            assert not truncated
            if step < 47:
                assert obs[3] * 10 == pytest.approx(2 + (index + 1) / 100)
                assert obs[-1] == index + 1
    with pytest.raises(ValueError, match='ended'):
        env.step(np.zeros(1))


def test_default_reset_progression_and_wrap(data_files):
    env = create_env(data_files)
    for day in (0, 1, 2, 0):
        obs, _ = env.reset()
        assert env.get_step_index() == day * 48
        for _ in range(48):
            env.step(np.zeros(1))


@pytest.mark.parametrize('date', ['2025-07-08', '2025-07-12', '2025-07-09 00:15', 'NaT'])
def test_invalid_selected_date(data_files, date):
    env = create_env(data_files)
    with pytest.raises(ValueError, match='date'):
        env.reset(start_date=date)


def test_partial_episode_rejected_before_resetting_state(data_files):
    env = create_env(data_files)
    env.reset()
    env.pcs.storage_units[0].reset(initial_level=25)
    with pytest.raises(ValueError, match='episode'):
        env.reset(start_date='2025-07-11 12:00')
    assert env.pcs.get_total_storage() == 25
    assert env.current_datetime == env.test_start_date


@pytest.mark.parametrize('index', [0, 22, 143])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_demand_rejected(data_files, index, value):
    set_load(data_files[0], index, value)
    with pytest.raises(ValueError, match='Consumption'):
        create_env(data_files)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unordered', 'invalid'])
def test_bad_demand_timestamps(data_files, mutation):
    df = pd.read_csv(data_files[0])
    if mutation == 'missing':
        df = df.drop(index=22)
    elif mutation == 'duplicate':
        df.loc[22, 'Datetime'] = df.loc[21, 'Datetime']
    elif mutation == 'unordered':
        df = df.iloc[::-1]
    else:
        df.loc[22, 'Datetime'] = 'invalid'
    df.to_csv(data_files[0], index=False)
    with pytest.raises(ValueError, match='timestamp'):
        create_env(data_files)


@pytest.mark.parametrize('column', ['predicted_consumption', 'feature'])
@pytest.mark.parametrize('value', [np.nan, np.inf, 'invalid'])
def test_invalid_predictions_and_features(data_files, column, value):
    df = pd.read_csv(data_files[1]).astype({column: object})
    df.loc[22, column] = value
    df.to_csv(data_files[1], index=False)
    with pytest.raises(ValueError, match='finite'):
        create_env(data_files)


@pytest.mark.parametrize('mutation', ['shift', 'missing', 'duplicate', 'unordered'])
def test_prediction_alignment(data_files, mutation):
    df = pd.read_csv(data_files[1], parse_dates=['timestamp'])
    if mutation == 'shift':
        df['timestamp'] += pd.Timedelta(minutes=30)
    elif mutation == 'missing':
        df = df.drop(index=22)
    elif mutation == 'duplicate':
        df.loc[22, 'timestamp'] = df.loc[21, 'timestamp']
    else:
        df = df.iloc[::-1]
    df.to_csv(data_files[1], index=False)
    with pytest.raises(ValueError, match='timestamp'):
        create_env(data_files)


def test_interpolation_preserves_source_and_records_changes(data_files):
    data_file, _ = data_files
    set_load(data_file, 22, np.nan)
    original_bytes = data_file.read_bytes()
    partitions = {'train': ('2025-07-09', '2025-07-11'),
                  'evaluation': ('2025-07-11', '2025-07-12')}
    env = create_env(data_files, missing_data_policy='interpolate', data_partitions=partitions)
    assert env.test_df['Consumption'].iloc[22] == pytest.approx(1.22)
    assert env.data_audit['source_sha256'] == hashlib.sha256(original_bytes).hexdigest()
    assert env.data_audit['changes'] == [
        {'timestamp': '2025-07-09T11:00:00', 'original': None, 'value': pytest.approx(1.22)}
    ]
    assert data_file.read_bytes() == original_bytes
    env.reset()
    for _ in range(48):
        obs, reward, _, _, _ = env.step(np.zeros(1))
        assert np.isfinite(obs).all() and np.isfinite(reward)


@pytest.mark.parametrize('index', [0, 47, 48, 143])
def test_interpolation_cannot_extrapolate_or_cross_partitions(data_files, index):
    set_load(data_files[0], index, np.nan)
    partitions = {'train': ('2025-07-09', '2025-07-10'),
                  'evaluation': ('2025-07-10', '2025-07-12')}
    with pytest.raises(ValueError, match='missing'):
        create_env(data_files, missing_data_policy='interpolate', data_partitions=partitions)


@pytest.mark.parametrize('partitions', [None,
    {'train': ('2025-07-09', '2025-07-10')},
    {'train': ('2025-07-09', '2025-07-11'), 'test': ('2025-07-10', '2025-07-12')},
])
def test_interpolation_requires_complete_nonoverlapping_partitions(data_files, partitions):
    with pytest.raises(ValueError, match='partition'):
        create_env(data_files, missing_data_policy='interpolate', data_partitions=partitions)


def test_direct_csv_lookup_uses_same_origin_and_rejects_wrapping(data_files):
    df = pd.read_csv(data_files[0], parse_dates=['Datetime'])
    df['Datetime'] += pd.Timedelta(hours=6)
    df.to_csv(data_files[0], index=False)
    dynamics = CSV_DataConsumptionDynamics({'data_file': data_files[0], 'data_mode': 'corrected'})
    assert dynamics.get_value(time=0) == 1
    assert dynamics.get_value(time=1 + 22 / 48) == pytest.approx(1.70)
    for time in (-1, 3, 0.01, np.nan):
        with pytest.raises(ValueError, match='Consumption'):
            dynamics.get_value(time=time)


def test_legacy_mode_preserves_first_day_and_missing_value(data_files):
    set_load(data_files[0], 22, np.nan)
    env = PCSEnv(test_data_file=data_files[0], predictions_file=data_files[1],
                 verbosity=0, data_mode='legacy')
    obs, _ = env.reset(start_date=pd.Timestamp('2025-07-10'))
    assert obs[1] * 10 == pytest.approx(1)
    dynamics = env.pcs.consumption_units[0].dynamics
    assert np.isnan(dynamics.get_value(time=22 / 48))
    assert dynamics.get_value(time=3) == 1


def test_alternating_environment_uses_selected_day(data_files):
    from energy_net.gym_envs.alternating_env import AlternatingISOEnv

    class ConstantPCSModel:
        def predict(self, observation, deterministic=True):
            return np.zeros(1), None

    pcs_env = create_env(data_files)
    env = AlternatingISOEnv(actual_csv=data_files[0], predicted_csv=data_files[1],
                            pcs_env=pcs_env, pcs_model=ConstantPCSModel(),
                            iso_verbosity=0, pcs_verbosity=0)
    env.reset()
    action = np.concatenate([np.full(48, 0.5), np.zeros(48)])
    env.step(action)
    env.step(action)
    assert pcs_env.get_step_index() == 96
    assert pcs_env.pcs.get_consumption() == pytest.approx(1.95)


def test_same_date_reset_refreshes_prices(data_files):
    from energy_net.grid_entities.management.price_curve import ActionBasedPriceStrategy

    env = create_env(data_files)
    env.reset()
    env.set_price_strategy(ActionBasedPriceStrategy(np.full(96, 0.5)))
    obs, _ = env.reset(start_date=env.test_start_date)
    assert obs[2] == pytest.approx(0.1)


def test_multi_day_episode_and_terminal_mask(data_files):
    env = create_env(data_files, episode_length_days=2)
    env.reset(start_date='2025-07-09')
    for step in range(96):
        obs, _, terminated, _, info = env.step(np.zeros(1))
        assert info['consumption_units'] == pytest.approx(1 + step / 100)
        assert info['current_price'] == pytest.approx((step % 48) / 100)
    assert terminated
    assert not obs[3:3 + 2 * env.prediction_horizon].any()


def test_nonfinite_price_is_rejected(data_files):
    class InvalidPriceStrategy(PriceCurveStrategy):
        def calculate_price(self, observation=None):
            return np.full(48, np.nan)

    env = create_env(data_files)
    env.set_price_strategy(InvalidPriceStrategy())
    with pytest.raises(ValueError, match='Price curves'):
        env.reset()


def test_policy_and_schema_validation(data_files):
    with pytest.raises(ValueError, match='policy'):
        load_demand_data(data_files[0], missing_data_policy='zero')
    df = pd.read_csv(data_files[0]).drop(columns='Consumption')
    df.to_csv(data_files[0], index=False)
    with pytest.raises(ValueError, match='Consumption'):
        load_demand_data(data_files[0])


def test_audit_covers_every_selected_sample(data_files):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / 'scripts/audit_demand_data.py'
    spec = importlib.util.spec_from_file_location('audit_demand_data', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.audit_demand_data(*data_files, ['2025-07-09', '2025-07-11'])
    assert report['selected_sample_count'] == 96
    assert report['distinct_load_profiles'] == 2
    assert report['distinct_price_profiles'] == 1
    assert report['preprocessing']['changes'] == []
    assert len(report['code_sha256']) == 4
    with pytest.raises(ValueError, match='nonoverlapping'):
        module.audit_demand_data(*data_files, ['2025-07-09', '2025-07-09'])
