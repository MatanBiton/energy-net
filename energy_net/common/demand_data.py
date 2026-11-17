"""Validated demand data and explicit, partition-local preprocessing."""

import hashlib

import numpy as np
import pandas as pd


def validate_time_index(index, dt, label):
    """Require a nonempty, ordered, unique index with the configured cadence."""
    if not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans:
        raise ValueError(f"{label} requires finite timestamps and at least one sample")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f"{label} timestamps must be ordered and unique")
    interval = pd.Timedelta(days=dt)
    if interval <= pd.Timedelta(0):
        raise ValueError("Demand timestep must be positive")
    if len(index) > 1 and not ((index[1:] - index[:-1]) == interval).all():
        raise ValueError(f"{label} has missing timestamps or an incorrect cadence")


def load_demand_data(data_file, dt=0.5 / 24, missing_data_policy='error',
                     data_partitions=None):
    """
    Return a validated copy of a demand CSV and its preprocessing audit.

    Policies are 'error' and 'interpolate'. Interpolation is linear in time,
    restricted to interior gaps within explicitly named, half-open partitions.
    Partitions must cover every row exactly once. Raw CSVs are never changed.
    """
    if missing_data_policy not in ('error', 'interpolate'):
        raise ValueError("Demand missing_data_policy must be 'error' or 'interpolate'")
    with open(data_file, 'rb') as source:
        source_hash = hashlib.sha256(source.read()).hexdigest()
    df = pd.read_csv(data_file, parse_dates=['Datetime'])
    if 'Consumption' not in df.columns:
        raise ValueError("Demand CSV must contain 'Datetime' and 'Consumption' columns")
    df = df.set_index('Datetime')
    validate_time_index(df.index, dt, 'Consumption data')
    try:
        df['Consumption'] = pd.to_numeric(df['Consumption'], errors='raise').astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("Consumption values must be numeric and finite") from error
    original = df['Consumption'].copy()
    if np.isinf(original.to_numpy()).any():
        raise ValueError("Consumption contains infinite values")

    partitions = []
    coverage = np.zeros(len(df), dtype=int)
    for name, bounds in (data_partitions or {}).items():
        start, end = [pd.Timestamp(value) for value in bounds]
        if start >= end:
            raise ValueError(f"Consumption partition {name} must have start before end")
        mask = (df.index >= start) & (df.index < end)
        coverage += mask.astype(int)
        partitions.append((name, start, end, mask))
    if data_partitions and not (coverage == 1).all():
        raise ValueError("Consumption partitions must cover each timestamp exactly once")

    if missing_data_policy == 'interpolate':
        if not partitions:
            raise ValueError("Consumption interpolation requires explicit data_partitions")
        for name, start, end, mask in partitions:
            values = df.loc[mask, 'Consumption']
            df.loc[mask, 'Consumption'] = values.interpolate(method='time', limit_area='inside')
    if not np.isfinite(df['Consumption'].to_numpy()).all():
        missing = df.index[df['Consumption'].isna()]
        raise ValueError(f"Consumption contains missing/non-finite values at {list(missing[:5])}")

    changes = [
        {'timestamp': timestamp.isoformat(), 'original': None,
         'value': float(df.loc[timestamp, 'Consumption'])}
        for timestamp in df.index[original.isna()]
    ]
    audit = {
        'source_sha256': source_hash,
        'missing_data_policy': missing_data_policy,
        'dt_days': dt,
        'sample_count': len(df),
        'partitions': [
            {'name': name, 'start': start.isoformat(), 'end': end.isoformat()}
            for name, start, end, mask in partitions
        ],
        'changes': changes,
    }
    return df, audit
