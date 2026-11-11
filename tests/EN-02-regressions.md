# EN-02 regression cases

These tests exercise the real PCSEnv, PCSUnit, battery and CSV consumption path
using temporary synthetic half-hourly CSVs and deterministic price curves.
They do not modify research data or production simulation code. Quantities are
PCSEnv internal units; no kW/kWh conversion is implied.

Run from the repository root using the project environment:

```sh
conda run -n en python -m pytest -q -rx tests/unittests/test_en02_regressions.py
```

The baseline result is 13 passed and 5 expected failures. The expected failures
are strict: an unexpected pass fails the suite so its marker must be reviewed
when the relevant repair or modeling decision lands. Only assertion failures
are expected; unrelated exceptions still fail normally.

To expose the original failures directly:

```sh
conda run -n en python -m pytest -q --runxfail --tb=short tests/unittests/test_en02_regressions.py
```

That diagnostic command deliberately exits with failure: 5 failed, 13 passed.

## Confirmed reproductions and conditional expectations

- Shortage: starting storage 5, zero action and demand 3.19168 leaves 1.80832;
  the legacy flag is true although the battery supplied the full demand. The
  same discrepancy occurs at exactly enough storage (3 for demand 3). These
  tests assume shortage means battery-uncovered demand. A reserve interpretation
  is a separate modeling decision, pending EN-03; these are not approved repairs.
- Date: selecting the second day returns first-day consumption at both reset
  and step. The CSV dynamics itself selects both days correctly; environment
  wiring passes episode-relative time without the selected date offset.
- Missing data: a NaN at the existing timestamp for step 22 propagates into the
  observation. The test allows explicit data rejection or finite handling; it
  does not select an interpolation policy. Prediction values are finite, which
  isolates the consumption path. Raw research CSVs remain untouched.

## Behavior protected by passing tests

- Zero demand, genuinely uncovered demand, and sufficient reserve boundaries.
- Explicit battery action followed by automatic demand balancing; the returned
  transaction is distinct from the total battery change.
- Capacity clipping near full/empty storage; money uses the realized transaction.
- Reset clears physical state and financial/episode counters when an explicit
  start date is supplied. Default date progression is not redefined.
- A complete 48-step day uses each half-hourly load and price in order, advances
  timestamps by 30 minutes, and terminates on the last step without truncation.

## Accounting deferred to the advisor

Preserve the legacy case: initial storage 0, action +3, demand 5, price 0.1.
The transaction is 3, final storage 0, monetary reward -0.3, penalty 1 and total
reward -1.3. No full-grid-settlement expectation is imposed. The user requested
leaving this unchanged pending the advisor meeting on November 12, 2026.

The residual two units would cost another 0.2 under full grid settlement, but
whether they are imported or unserved and what penalty applies remains open.

## Scope

No training, solver runs, reward repairs, terminal storage targets, PV extensions,
efficiency changes or new tariff/netting rules are introduced. Historical result
archives are not needed to run these minimal synthetic cases.
