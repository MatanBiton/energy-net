"""Audit selected half-hour demand and price profiles without modifying raw CSVs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

# Allow execution from a checkout without installing the package again.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from energy_net.gym_envs.pcs_env import PCSEnv


def audit_demand_data(data_file, predictions_file, start_dates, days=1,
                      missing_data_policy='error', data_partitions=None):
    """Check every selected sample and report actual profile identities."""
    if days < 1:
        raise ValueError("Audit days must be positive")
    dates = [pd.Timestamp(date) + pd.Timedelta(days=day)
             for date in start_dates for day in range(days)]
    if not dates or len(set(dates)) != len(dates):
        raise ValueError("Audit requires nonempty, nonoverlapping selected dates")
    env = PCSEnv(test_data_file=data_file, predictions_file=predictions_file,
                 data_mode='corrected', missing_data_policy=missing_data_policy,
                 data_partitions=data_partitions, verbosity=0)
    profiles = []
    try:
        for date in dates:
            obs, _ = env.reset(start_date=date)
            loads, prices = [], []
            for step in range(env.max_steps):
                obs, reward, terminated, truncated, info = env.step(np.zeros(1))
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise ValueError(f"Non-finite observation or reward at {env.current_datetime}")
                timestamp = date + pd.Timedelta(minutes=30 * step)
                expected = env.test_df.loc[timestamp, 'Consumption']
                if info['consumption_units'] != expected:
                    raise ValueError(f"Demand lookup disagrees with selected timestamp {timestamp}")
                loads.append(info['consumption_units'])
                prices.append(info['current_price'])
            profiles.append({
                'start_date': date.isoformat(),
                'sample_count': len(loads),
                'total_load': float(np.sum(loads)),
                'mean_load': float(np.mean(loads)),
                'load_sha256': hashlib.sha256(np.asarray(loads, dtype='<f8').tobytes()).hexdigest(),
                'price_sha256': hashlib.sha256(np.asarray(prices, dtype='<f8').tobytes()).hexdigest(),
            })
    finally:
        env.close()
    try:
        revision = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip())
    except subprocess.CalledProcessError:
        revision = None
        dirty = None
    with open(predictions_file, 'rb') as source:
        prediction_hash = hashlib.sha256(source.read()).hexdigest()
    return {
        'code_revision': revision,
        'working_tree_dirty': dirty,
        'code_sha256': {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ('energy_net/common/demand_data.py',
                         'energy_net/gym_envs/pcs_env.py',
                         'energy_net/grid_entities/consumption/consumption_dynamics.py',
                         'scripts/audit_demand_data.py')
        },
        'data_mode': 'corrected',
        'price_strategy': 'SineWavePriceStrategy',
        'prediction_sha256': prediction_hash,
        'preprocessing': env.data_audit,
        'selected_sample_count': sum(profile['sample_count'] for profile in profiles),
        'distinct_load_profiles': len({profile['load_sha256'] for profile in profiles}),
        'distinct_price_profiles': len({profile['price_sha256'] for profile in profiles}),
        'profiles': profiles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-file', required=True)
    parser.add_argument('--predictions-file', required=True)
    parser.add_argument('--start-date', action='append', default=[])
    parser.add_argument('--date-range', action='append', default=[], metavar='START,END',
                        help='Select daily starts in a half-open date range')
    parser.add_argument('--days', type=int, default=1)
    parser.add_argument('--missing-data-policy', choices=['error', 'interpolate'], default='error')
    parser.add_argument('--partition', action='append', default=[], metavar='NAME,START,END',
                        help='Declare a data partition; start inclusive, end exclusive')
    parser.add_argument('--output', help='Write audit JSON here; otherwise print it')
    args = parser.parse_args()
    partitions = {}
    for partition in args.partition:
        fields = partition.split(',')
        if len(fields) != 3 or fields[0] in partitions:
            parser.error('Each partition must have a unique NAME,START,END')
        partitions[fields[0]] = fields[1:]
    start_dates = list(args.start_date)
    for date_range in args.date_range:
        fields = date_range.split(',')
        if len(fields) != 2:
            parser.error('Each date range must have START,END')
        start_dates.extend(pd.date_range(fields[0], fields[1], freq='D', inclusive='left'))
    if not start_dates:
        parser.error('Select at least one --start-date or --date-range')
    report = audit_demand_data(args.data_file, args.predictions_file, start_dates,
                               args.days, args.missing_data_policy, partitions or None)
    content = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        Path(args.output).write_text(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
