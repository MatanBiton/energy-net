# EN-04 demand data

## Selecting the data path

Existing callers keep `data_mode='legacy'` by default so historical runs remain
selectable. This retains episode-relative demand, implicit CSV interpolation and
wrapping, and does not provide finite-data guarantees.

Use `data_mode='corrected'` for EN-04. The default corrected policy is
`missing_data_policy='error'`. Demand, predictions and features must have finite
numeric values and identical, ordered, unique timestamp grids at the configured
cadence. Missing rows are rejected rather than silently manufactured. Selecting a
date outside that grid, or an episode without enough demand samples, raises a
`ValueError` before resetting physical state.

```python
from energy_net.gym_envs.pcs_env import PCSEnv

# For a complete, finite pair of demand and prediction files:
env = PCSEnv(
    test_data_file='demand.csv',
    predictions_file='predictions.csv',
    data_mode='corrected',
    verbosity=0,
)
obs, info = env.reset(start_date='2025-07-10')
```

Corrected reset and step use the selected date's sample offset from the CSV
origin. Prediction and feature windows use the same exact index. Demand queries
do not wrap past the dataset or interpolate off-grid times. A terminal observation
retains the existing last-consumed-demand convention and zero prediction mask;
no extra demand sample is consumed after termination. Prediction/feature padding
at the dataset end retains the existing last-value convention for price inputs.
Reset without an explicit date advances from the current timestamp and wraps to
the dataset start after the final sample; an incomplete remaining episode is
rejected. Price caches are refreshed on reset and strategy replacement.

## Explicit interpolation

`missing_data_policy='interpolate'` fills existing NaNs using linear interpolation
in timestamp coordinates. It fills interior gaps only, with no extrapolation and
no maximum gap length. Infinite or nonnumeric values and missing timestamps are
always rejected. Unfilled gaps are errors.

Interpolation requires named `data_partitions` covering each row exactly once.
Each `(start, end)` interval includes its start and excludes its end. Adjacent
partitions never exchange values; a missing endpoint cannot borrow a value from
the neighboring partition. Declare actual training/evaluation boundaries rather
than labeling the combined dataset as one partition. The loader validates the
declared boundaries; it cannot infer how an external experiment splits data.

```python
partitions = {
    'train': ('2025-07-09', '2025-10-07'),
    'unused': ('2025-10-07', '2025-10-27'),
    'evaluation': ('2025-10-27', '2025-11-06'),
    'remaining': ('2025-11-06', '2025-12-02'),
}
env = PCSEnv(
    data_mode='corrected',
    missing_data_policy='interpolate',
    data_partitions=partitions,
    verbosity=0,
)
```

Only the in-memory realized-demand copy is repaired. Raw CSVs and prediction
files are untouched. Interpolated realized demand is not passed to predictor
training or used to regenerate forecasts. This is offline trace preprocessing,
not a causal forecasting method. `env.data_audit` records the raw file SHA-256,
policy, partition boundaries, and every changed timestamp and value.

The standalone `CSV_DataConsumptionDynamics` accepts the same settings in its
`params` dictionary; its default remains legacy for existing callers.

## Reproducing the selected-sample audit

From the repository root, using the `en` environment:

```sh
conda run -n en python scripts/audit_demand_data.py \
  --data-file tests/gym/data_for_tests/synthetic_household_consumption_test.csv \
  --predictions-file tests/gym/data_for_tests/consumption_predictions.csv \
  --missing-data-policy interpolate \
  --partition train,2025-07-09,2025-10-07 \
  --partition unused,2025-10-07,2025-10-27 \
  --partition evaluation,2025-10-27,2025-11-06 \
  --partition remaining,2025-11-06,2025-12-02 \
  --date-range 2025-07-09,2025-10-07 \
  --date-range 2025-10-27,2025-11-06 \
  --output docs/en-04-demand-audit.json
```

The selection corresponds to day indices 0–89 and 110–119 relative to July 9,
2025. The script checks each realized lookup against the selected timestamp and
checks finite observations and rewards. It reports daily load and price hashes,
profile counts, preprocessing changes, code/data hashes, and checkout revision
and modification state. Dates must not overlap. Prices use the existing default
`SineWavePriceStrategy` and actions are zero; this is a data audit, not training
or a controller-performance comparison.

The current CSVs reproduce the reported July 14, 2025, 11:00 repair to
`4.302115`. All 4,800 selected samples pass, with 100 distinct daily demand
profiles and one price profile. The ten selected evaluation days have ten
distinct demand profiles. There are 13 missing values across the complete test
CSV, but only one within these selected dates. The explicit interpolation policy
repairs all 13 within their respective partitions and records all of them.

This verifies the reported preprocessing value and selected-sample checks against
the current CSVs. It does not recover the original experiment archives or certify
their full historical environment. See the existing baseline manifest for that
remaining provenance limitation.

## Validation and scope

```sh
conda run -n en python -m pytest -q tests/unittests
```

Tests cover every half-hour in multiple distinct days, finite inputs, prediction
alignment, partition boundaries, unmodified raw data, terminal handling, default
date progression, and alternating ISO/PCS date synchronization. The EN-02 date
and missing-value cases now pass in corrected mode. Its two shortage expectations
remain strict expected failures pending the separate modeling decision.

Battery transitions, shortage semantics, reward settlement, forecasting models,
and tariffs are unchanged. Legacy behavior remains covered separately.
