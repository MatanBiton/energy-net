# EN-05: corrected single-PCS settlement

Implemented 16 November 2026 against the settlement decision:
https://app.notion.com/p/417a724f2ce881679217ed10e8b98387

Select both `data_mode='corrected'` and `settlement_mode='corrected'`.
Settlement defaults to `legacy` to preserve EN-02 and historical comparisons.
This is independent of EN-04's data selection switch.

```python
env = PCSEnv(data_mode='corrected', settlement_mode='corrected',
             max_purchase_price=10.0, max_sale_price=10.0, verbosity=0)
obs, info = env.reset(start_date='2025-07-10')
```

The example upper bounds are configurable internal price units, not a calibrated
market limit. Both must be finite and nonnegative. Each day's 48 prices must
satisfy 0 <= sale <= purchase and both configured bounds. ISO sell maps to PCS
purchase; ISO buy maps to PCS sale. Curves use midnight-aligned inputs and
calendar slots. Reset and strategy changes invalidate the cache; after replacing
a strategy, refresh the observation before the next policy decision.

The corrected observation always contains purchase and sale in positions 2 and 3,
replacing the old single-price channel. Subsequent channels shift by one;
existing trained policies need a matching observation adapter or retraining.
Terminal observations retain the final settled price pair, since no further
policy decision is made.

The two-stage battery update remains unchanged. PCSUnit returns realized action
q after saturation and before automatic demand balancing. Battery service is
S + q - S_next, residual demand D is L minus battery service, and net meter g is
q + D. Cost is purchase*max(g,0) - sale*max(-g,0). Money and reward are -cost.
No extra interval multiplier, legacy shortage penalty, reserve obligation,
terminal payment, or second charge for q is applied. Unserved load is zero.
Requests outside [-10,10] or nonfinite/non-scalar-array inputs are rejected
before changing state; action_scale must be 1 and intervals half-hourly.

Info separates realized_action_units (q), battery_service_units,
residual_demand_units (battery-uncovered, grid-served), unserved_energy_units,
grid_exchange_units, grid_import_units, grid_export_units, purchase_price,
sale_price, net_electricity_cost, step_money, step_penalty, reserve_penalty,
and counterparty_receipt. Existing transaction and cumulative money fields
remain available; shortage_count and total_shortage_penalty stay zero in
corrected settlement.

Validation: all six contract examples, upper/lower saturation, zero demand,
export, negative action with positive net import, asymmetric netting, invalid
tariffs/actions, calendar rollover, observed/settled tariffs, strategy changes,
reset and 96 seeded transitions with energy/cash invariants. The required example
returns -0.5; the legacy case remains -1.3 in the separate EN-02 suite.

Scope: this completes the single-PCS EN-05 settlement repair. It does not claim
implementation of the entire EN-03 MDP: previous-demand reset, seven-feature
selection, explicit-only date scheduling and full future-episode price
prevalidation remain separate integration work. Community clearing/transfers,
ISO learning, oracle adaptation (EN-06) and pilot training (EN-08) are not built
by this change. Daily tariffs are validated when generated; invalid future
curves may reject a later transition/observation rather than reset.

Delivery is an uncommitted local repair based on research HEAD
2eb399de721a03e8789f70106a8cb9a5197101e3 plus existing EN-04 working changes.
Raw CSVs and legacy characterization are preserved.
