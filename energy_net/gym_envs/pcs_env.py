import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from energy_net.common.demand_data import validate_time_index
from energy_net.grid_entities.consumption.consumption_dynamics import CSV_DataConsumptionDynamics
from energy_net.grid_entities.PCSUnit.pcs_unit import PCSUnit
from energy_net.grid_entities.management.price_curve import PriceCurveStrategy, SineWavePriceStrategy
from energy_net.grid_entities.storage.battery_dynamics import DeterministicBattery
from energy_net.grid_entities.storage.battery import Battery
from energy_net.grid_entities.consumption.consumption_unit import ConsumptionUnit
from energy_net.foundation.model import State, Action
from datetime import timedelta
import numpy as np
import os


class PCSEnv(gym.Env):
    """
    Simplified unit-less Gym environment for PCSUnit.
    - Uses internal "units" everywhere (battery capacity 0..100, consumption ~2 units/step).
    - Actions are normalized in [-1, 1] and passed directly to PCS/Battery.
    - No unit conversions to kWh/kW — everything stays in the same internal units.
    - Loads pre-computed predictions from CSV file.
    - Automatically detects and includes feature columns in observation space.
    """

    def __init__(self,
                 test_data_file=None,
                 predictions_file=None,
                 dt=0.5 / 24,  # 30 minutes in days
                 episode_length_days=1,
                 prediction_horizon=None,
                 shortage_penalty=1.0,
                 action_scale=1.0,
                 price_strategy: PriceCurveStrategy = None,
                 render_mode: str | None = None,
                 verbosity: int = 2,
                 data_mode='legacy',
                 missing_data_policy='error',
                 data_partitions=None,
                 settlement_mode='legacy',
                 max_purchase_price=10.0,
                 max_sale_price=10.0):
        """
        Initialize PCS environment.
        
        Note: Predictions are encoded with zero-padding after episode end
        plus a validity mask (masked_adaptive encoding) for optimal performance.
        
        Args:
            prediction_horizon: Number of future timesteps to include in observation.
                If None, defaults to max_steps (full day visibility).
            data_mode: 'legacy' preserves historical lookups; 'corrected' selects
                dated, validated demand and aligned prediction/feature inputs.
            missing_data_policy: Corrected demand policy, 'error' or 'interpolate'.
            settlement_mode: 'legacy' preserves old rewards; 'corrected' enables
                net-meter settlement and dual PCS-perspective price observations.
                Requires corrected data, half-hour intervals and action_scale=1.
            max_purchase_price, max_sale_price: Finite tariff bounds in internal
                price units (default 10 each); choose explicitly per experiment.
            data_partitions: Named (start, end) timestamp pairs, end exclusive.
                Required for interpolation; partitions cannot exchange values.
        """
        super().__init__()

        # Calculate project root for absolute paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Set default file paths if not provided
        if test_data_file is None:
            test_data_file = os.path.join(project_root, 'tests/gym/data_for_tests/synthetic_household_consumption_test.csv')
        if predictions_file is None:
            predictions_file = os.path.join(project_root, 'tests/gym/data_for_tests/consumption_predictions.csv')

        # -------------------------
        # Basic parameters
        # -------------------------
        if data_mode not in ('legacy', 'corrected'):
            raise ValueError("data_mode must be 'legacy' or 'corrected'")
        self.data_mode = data_mode
        if settlement_mode not in ('legacy', 'corrected'):
            raise ValueError("settlement_mode must be 'legacy' or 'corrected'")
        self.settlement_mode = settlement_mode
        self.max_purchase_price = max_purchase_price
        self.max_sale_price = max_sale_price
        if settlement_mode == 'corrected':
            if data_mode != 'corrected':
                raise ValueError("Corrected settlement requires corrected data_mode")
            if action_scale != 1.0:
                raise ValueError("Corrected settlement requires action_scale=1")
            if not all(np.isfinite(x) and x >= 0 for x in
                       (max_purchase_price, max_sale_price)):
                raise ValueError("Tariff upper bounds must be finite and nonnegative")
            if not np.isclose(dt, 0.5 / 24):
                raise ValueError("Corrected settlement requires half-hour intervals")
        self._settled_prices = None
        if data_mode == 'corrected':
            if not np.isfinite(dt) or dt <= 0 or not np.isfinite(episode_length_days) or episode_length_days <= 0:
                raise ValueError("Timestep and episode length must be positive and finite")
            if not np.isclose(episode_length_days / dt, round(episode_length_days / dt)):
                raise ValueError("Episode length must contain a whole number of timesteps")
        self.dt = dt
        self.episode_length_days = episode_length_days
        self.max_steps = int(episode_length_days / dt)
        if self.data_mode == 'corrected':
            self.max_steps = int(round(episode_length_days / dt))
        
        # Default prediction_horizon to full day if not specified
        if prediction_horizon is None:
            prediction_horizon = self.max_steps
        self.prediction_horizon = prediction_horizon
        self.shortage_penalty = shortage_penalty
        self.action_scale = action_scale
        if price_strategy is None:
            self.price_strategy = SineWavePriceStrategy()
        else:
            self.price_strategy = price_strategy
        
        # Handle backward compatibility: render_mode='human' maps to verbosity=2
        if render_mode == 'human':
            self.verbosity = 2
        elif render_mode is None:
            self.verbosity = verbosity
        else:
            self.verbosity = verbosity
        
        self.render_mode = render_mode  # Keep for backward compatibility
        self.last_action = None
        self.cached_day_prices = None
        self.last_price_update_date = None

        # -------------------------
        # Load test data
        # -------------------------
        self.data_audit = None
        if self.data_mode == 'corrected':
            consumption_dynamics = CSV_DataConsumptionDynamics(params={
                'data_file': test_data_file, 'data_mode': self.data_mode, 'dt': dt,
                'missing_data_policy': missing_data_policy, 'data_partitions': data_partitions,
            })
            self.test_df = consumption_dynamics.consumption_data.copy()
            self.data_audit = consumption_dynamics.data_audit
        else:
            self.test_df = pd.read_csv(
                test_data_file,
                index_col=0,
                parse_dates=True
            )

        self.test_start_date = self.test_df.index[0]
        self.test_end_date = self.test_df.index[-1]

        # -------------------------
        # Load predictions and automatically detect features
        # -------------------------
        pred_df = pd.read_csv(
            predictions_file,
            parse_dates=['timestamp']
        ).set_index('timestamp')
        if self.data_mode == 'corrected':
            validate_time_index(pred_df.index, dt, 'Predictions')
            if not pred_df.index.equals(self.test_df.index):
                raise ValueError("Prediction timestamps must match consumption timestamps")
            if 'predicted_consumption' not in pred_df.columns:
                raise ValueError("Predictions require predicted_consumption")
            try:
                pred_df = pred_df.apply(pd.to_numeric, errors='raise').astype(float)
            except (TypeError, ValueError) as error:
                raise ValueError("Predictions and features must be numeric and finite") from error
            if not np.isfinite(pred_df.to_numpy()).all():
                raise ValueError("Predictions and features contain non-finite values")
        else:
            pred_df = pred_df.sort_index()

        self.pred_df = pred_df
        self.prediction_cache = pred_df['predicted_consumption'].values
        self.prediction_timestamps = pred_df.index

        # Automatically detect feature columns (everything except timestamp and predicted_consumption)
        excluded_cols = {'predicted_consumption'}
        self.feature_columns = [col for col in pred_df.columns if col not in excluded_cols]
        self.num_features = len(self.feature_columns)

        if self.verbosity > 0:
            print(f"Loaded {len(self.prediction_cache)} aligned predictions")
            print(f"Detected {self.num_features} feature columns: {self.feature_columns}")

        # Cache feature values for fast lookup
        if self.num_features > 0:
            self.feature_cache = pred_df[self.feature_columns].values
        else:
            self.feature_cache = None

        # ==============================================================
        # Battery
        # ==============================================================
        battery_dynamics = DeterministicBattery(model_parameters={})
        battery_config = {
            "min": 0.0,
            "max": 100.0,
            "init": 0.0
        }
        battery = Battery(
            dynamics=battery_dynamics,
            config=battery_config
        )

        # ==============================================================
        # Consumption unit (test data)
        # ==============================================================
        if self.verbosity > 0:
            print("Creating consumption unit on testing data...")
        if self.data_mode == 'legacy':
            consumption_dynamics = CSV_DataConsumptionDynamics(params={'data_file': test_data_file})
        consumption_unit_config = {
            'data_file': test_data_file,
            'consumption_capacity': 6.0
        }
        consumption_unit = ConsumptionUnit(
            dynamics=consumption_dynamics,
            config=consumption_unit_config
        )

        self.pcs = PCSUnit(
            storage_units=[battery],
            consumption_units=[consumption_unit]
        )

        # ==============================================================
        # Gym spaces
        # ==============================================================
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation space uses masked_adaptive encoding:
        # zero-padded predictions + validity mask + battery capacity constraints
        encoded_prediction_dim = prediction_horizon * 2
        
        # Observation space:
        # [storage, consumption, price] + [zero-padded predictions + mask] + [battery capacities] + [features]
        obs_dim = 3 + encoded_prediction_dim + 2 + self.num_features
        if self.settlement_mode == 'corrected':
            obs_dim += 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        if self.verbosity > 0:
            print(f"Observation space dimension: {obs_dim}")
            print("  - Base features (storage, consumption, price): 3")
            print(f"  - Zero-padded predictions + validity mask: {encoded_prediction_dim}")
            print("  - Battery capacity constraints: 2")
            print(f"  - Additional time features: {self.num_features}")

        # ==============================================================
        # Episode state
        # ==============================================================
        self.current_step = 0
        self.start_date = self.test_start_date
        self.current_datetime = self.start_date

        self.total_reward = 0.0
        self.total_money_earned = 0.0
        self.total_shortage_penalty = 0.0
        self.shortage_count = 0

    def get_money(self):
        """Returns the accumulated money made from transactions, ignoring shortage penalties."""
        return self.total_money_earned

    def _get_current_price(self, energy_flow_direction=None) -> float:
        """
        Get the current price, optionally using asymmetric buy/sell pricing.
        
        Args:
            energy_flow_direction: If provided, 'buy' means PCS is buying (charging),
                'sell' means PCS is selling (discharging). If None, uses symmetric pricing.
        
        Returns:
            float: The price for the current timestep
        """
        # Only calculate the full day curves if the date has changed
        if self.cached_day_prices is None or self.current_datetime.date() != self.last_price_update_date:
            # ISO model always expects 48 timesteps (full day), regardless of PCS prediction_horizon
            iso_horizon = 48
            # Corrected daily commitments use midnight-aligned inputs and slots.
            price_time = self.current_datetime
            try:
                if self.settlement_mode == 'corrected':
                    self.current_datetime = pd.Timestamp(price_time).normalize()
                predicted_consumption = self._get_predicted_consumption(iso_horizon)
                day_features = self._get_feature_window(iso_horizon)
            finally:
                self.current_datetime = price_time
            iso_input = np.concatenate([predicted_consumption / 10.0, day_features])

            # Call the ISO brain ONLY ONCE per day for all price curves
            # Check if strategy supports asymmetric pricing
            if hasattr(self.price_strategy, 'use_asymmetric_pricing') and self.price_strategy.use_asymmetric_pricing:
                # Cache both buy and sell prices
                self.cached_buy_prices = self.price_strategy.calculate_buy_price(iso_input)
                self.cached_sell_prices = self.price_strategy.calculate_sell_price(iso_input)
                # Also cache base prices for compatibility
                self.cached_day_prices = self.price_strategy.calculate_price(iso_input)
            else:
                # Symmetric pricing - same price for buy and sell
                self.cached_day_prices = self.price_strategy.calculate_price(iso_input)
                self.cached_buy_prices = None
                self.cached_sell_prices = None
            
            if self.data_mode == 'corrected':
                for curve in (self.cached_day_prices, self.cached_buy_prices, self.cached_sell_prices):
                    if curve is not None and (np.ndim(curve) != 1 or len(curve) == 0
                                              or not np.isfinite(curve).all()):
                        self.cached_day_prices = None
                        raise ValueError("Price curves must contain finite values")
                self.cached_price_start_index = self.get_step_index()
            if self.settlement_mode == 'corrected':
                purchase = self.cached_sell_prices if self.cached_sell_prices is not None else self.cached_day_prices
                sale = self.cached_buy_prices if self.cached_buy_prices is not None else self.cached_day_prices
                purchase, sale = np.asarray(purchase), np.asarray(sale)
                if (purchase.shape != (48,) or sale.shape != (48,)
                        or not np.isfinite(purchase).all() or not np.isfinite(sale).all()
                        or np.any(sale < 0) or np.any(purchase < sale)
                        or np.any(purchase > self.max_purchase_price)
                        or np.any(sale > self.max_sale_price)):
                    self.cached_day_prices = None
                    raise ValueError("Tariffs require 48 finite slots and 0 <= sale <= purchase within configured bounds")
            self.last_price_update_date = self.current_datetime.date()

        intra_day_step = self.current_step % self.max_steps
        if self.data_mode == 'corrected':
            intra_day_step = self.get_step_index() - self.cached_price_start_index
            for curve in (self.cached_day_prices, self.cached_buy_prices, self.cached_sell_prices):
                if curve is not None and intra_day_step >= len(curve):
                    raise ValueError("Price curve does not cover the selected timestep")
        
        if self.settlement_mode == 'corrected':
            intra_day_step = self.current_datetime.hour * 2 + self.current_datetime.minute // 30

        # Return appropriate price based on direction if asymmetric pricing is enabled
        if energy_flow_direction and self.cached_buy_prices is not None and self.cached_sell_prices is not None:
            if energy_flow_direction == 'buy':
                # PCS is buying (charging) - use sell price (ISO is selling to PCS)
                return float(self.cached_sell_prices[intra_day_step])
            elif energy_flow_direction == 'sell':
                # PCS is selling (discharging) - use buy price (ISO is buying from PCS)
                return float(self.cached_buy_prices[intra_day_step])
        
        # Default: return symmetric price
        return float(self.cached_day_prices[intra_day_step])
    def set_price_strategy(self, strategy: PriceCurveStrategy):
        """
        Explicitly updates the pricing logic by injecting a new
        PriceCurveStrategy object.
        """
        if not isinstance(strategy, PriceCurveStrategy):
            raise TypeError("The provided strategy must be an instance of PriceCurveStrategy.")

        self.price_strategy = strategy
        if self.data_mode == 'corrected':
            self.cached_day_prices = None
            self.last_price_update_date = None

    def _get_predicted_consumption(self, num_steps):
        """Get predicted consumption from preloaded cache."""
        current_idx = self.get_step_index()
        end_idx = current_idx + num_steps

        if end_idx > len(self.prediction_cache):
            available = self.prediction_cache[current_idx:]
            padding = np.full(num_steps - len(available), self.prediction_cache[-1])
            return np.concatenate([available, padding])

        return self.prediction_cache[current_idx:end_idx]

    def _get_current_features(self):
        """Get current feature values from preloaded cache."""
        if self.feature_cache is None or self.num_features == 0:
            return np.array([])

        current_idx = self.get_step_index()

        if current_idx >= len(self.feature_cache):
            return self.feature_cache[-1]

        return self.feature_cache[current_idx]

    def reset(self, start_date=None, options=None, seed=None):
        """
        Resets the environment to an initial state for a new episode.

        This method handles time synchronization, internal PCS unit resets,
        and resets all financial/performance tracking metrics.
        """
        if self.data_mode == 'corrected':
            selected_date = self.current_datetime if start_date is None else pd.Timestamp(start_date)
            if start_date is None and selected_date > self.test_end_date:
                selected_date = self.test_start_date
            index = self._get_data_index(selected_date)
            if index + self.max_steps > len(self.test_df):
                raise ValueError("Selected episode extends beyond consumption data")
            self.current_datetime = selected_date
            self.cached_day_prices = None
            self.last_price_update_date = None

        # 1. Standard Gym reset for seeding
        super().reset(seed=seed)

        # 2. Reset episode-specific counters
        self.current_step = 0
        self.total_reward = 0.0
        self.total_money_earned = 0.0
        self.total_shortage_penalty = 0.0
        self.shortage_count = 0

        # 3. Handle Time Synchronization
        # If a specific date is provided (e.g., during tandem training), use it.
        # Otherwise, check if we've hit the end of the test data and loop if necessary.
        if start_date is not None and self.data_mode == 'legacy':
            self.current_datetime = start_date
        else:
            if self.data_mode == 'legacy' and self.current_datetime >= self.test_end_date:
                print("Reached end of dataset. Looping back to start.")
                self.current_datetime = self.test_start_date

        # 4. Reset Physical Units
        # Resets the battery to empty (or a configured initial level) and clears PCS logs.
        self.pcs.reset(initial_storage_unit_level=0)
        self.last_action = None
        self._settled_prices = None

        # 5. Initialize consumption at timestep 0 without applying actions
        # This ensures consumption values are calculated before returning initial observation
        initial_time = self.get_step_index() * self.dt if self.data_mode == 'corrected' else 0.0
        initial_state = State({"time": initial_time})
        self.pcs.update(state=initial_state, actions=None)

        # 6. Return Initial Observation
        # Note: _get_obs() will internally query the price_strategy for the starting price.
        return self._get_obs(), {}

    def step(self, action):
        """
        Executes one 30-minute time step in the environment.

        Args:
            action: A NumPy array containing the normalized battery intent [-10, 10].

        Returns:
            obs: The next observation vector.
            reward: The net financial reward (Money earned - Shortage penalties).
            terminated: Whether the 24-hour day is complete.
            truncated: False (standard Gym API).
            info: A dictionary containing diagnostic information for the step.
        """
        if self.data_mode == 'corrected' and self.current_step >= self.max_steps:
            raise ValueError("Episode has ended; reset before stepping again")

        if self.settlement_mode == 'corrected':
            values = np.asarray(action)
            if values.shape != (1,) or not np.isfinite(values).all() or abs(float(values[0])) > 10:
                raise ValueError("Action must contain one finite value in [-10, 10]")
            # Validate and commit tariffs before mutating physical state.
            purchase_price = self._get_current_price('buy')
            sale_price = self._get_current_price('sell')

        # 1. Process the Agent's Action
        raw_action = float(action[0]) * self.action_scale
        self.last_action = raw_action

        # Record state before update for the info dict
        storage_before = self.pcs.get_total_storage()
        battery_entity = self.pcs.storage_units[0]

        # 2. Update the Physical Simulation
        # Convert step count to a relative time for the PCS logic
        current_time_relative = self.current_step * self.dt
        if self.data_mode == 'corrected':
            current_time_relative = self.get_step_index() * self.dt
        state = State()
        state.set_attribute('time', current_time_relative)

        # Apply action to the Battery via the PCS wrapper
        actions = {"Battery_0": Action({'value': raw_action})}
        energy_sold_or_bought = self.pcs.update(state, actions)

        # Record state after update
        storage_after = self.pcs.get_total_storage()
        consumption_units = float(self.pcs.get_consumption())

        # Determine if a shortage occurred (demand exceeded available discharge)
        try:
            available_discharge_units = float(battery_entity.get_available_discharge_capacity())
        except Exception:
            available_discharge_units = getattr(battery_entity, 'available_discharge', 0.0)

        if self.settlement_mode == 'corrected':
            # q is the actual clipped action returned by PCSUnit before balancing.
            battery_service = storage_before + energy_sold_or_bought - storage_after
            residual_demand = max(0.0, consumption_units - battery_service)
            grid_exchange = energy_sold_or_bought + residual_demand
            grid_import = max(grid_exchange, 0.0)
            grid_export = max(-grid_exchange, 0.0)
            net_cost = purchase_price * grid_import - sale_price * grid_export
            step_money = -net_cost
            # Residual energy is grid-served; no shortage or reserve cost.
            shortage = False
            step_penalty = 0.0
            current_price = purchase_price if grid_exchange >= 0 else sale_price
            self._settled_prices = (purchase_price, sale_price)

        else:
            # 3. Calculate Financials using the Price Strategy
            # Determine energy flow direction for asymmetric pricing
            if energy_sold_or_bought > 0:
                # Positive means battery charged (PCS bought energy from ISO)
                energy_flow = 'buy'
            elif energy_sold_or_bought < 0:
                # Negative means battery discharged (PCS sold energy to ISO)
                energy_flow = 'sell'
            else:
                # No energy flow
                energy_flow = None
        
            # Fetch the appropriate price based on energy flow direction
            current_price = self._get_current_price(energy_flow_direction=energy_flow)

            # Money earned: (Price * Energy Sold). If energy is bought, this is negative.
            step_money = current_price * -energy_sold_or_bought

            # 4. Handle Penalties
            shortage = consumption_units > available_discharge_units
            step_penalty = 0.0
            if shortage:
                step_penalty = self.shortage_penalty
                self.shortage_count += 1
                self.total_shortage_penalty += step_penalty

        # 5. Update Cumulative Metrics
        self.total_money_earned += step_money
        reward = step_money - step_penalty
        self.total_reward += reward

        # 6. Advance Environment Time
        self.current_step += 1
        self.current_datetime += timedelta(days=self.dt)

        # 7. Check for Episode Termination (typically 48 steps / 1 day)
        terminated = self.current_step >= self.max_steps
        truncated = False

        # 8. Prepare Return Values
        obs = self._get_obs()

        info = {
            'storage_before_units': storage_before,
            'storage_after_units': storage_after,
            'energy_sold_or_bought_units': energy_sold_or_bought,
            'consumption_units': consumption_units,
            'available_discharge_units': available_discharge_units,
            'shortage': shortage,
            'battery_action': raw_action,
            'step_money': step_money,
            'current_price': current_price,  # Added for better traceability
            'total_money_so_far': self.total_money_earned
        }

        if self.settlement_mode == 'corrected':
            info.update({
                'settlement_mode': 'corrected',
                'realized_action_units': energy_sold_or_bought,
                'battery_service_units': battery_service,
                'residual_demand_units': residual_demand,
                'unserved_energy_units': 0.0,
                'grid_exchange_units': grid_exchange,
                'grid_import_units': grid_import,
                'grid_export_units': grid_export,
                'purchase_price': purchase_price,
                'sale_price': sale_price,
                'net_electricity_cost': net_cost,
                'step_penalty': 0.0,
                'reserve_penalty': 0.0,
                'counterparty_receipt': net_cost,
            })
        return obs, reward, terminated, truncated, info

    def _get_feature_window(self, num_steps):
        """Get feature values for a range of timesteps starting from current datetime."""
        if self.feature_cache is None or self.num_features == 0:
            return np.array([])

        # Calculate starting index in the cache
        current_idx = self.get_step_index()
        end_idx = current_idx + num_steps

        if end_idx > len(self.feature_cache):
            # Handle end-of-dataset by padding with the last known features
            available = self.feature_cache[current_idx:]
            padding_needed = num_steps - len(available)
            padding = np.tile(self.feature_cache[-1], (padding_needed, 1))
            features = np.vstack([available, padding])
        else:
            features = self.feature_cache[current_idx:end_idx]

        # Flatten from (48, 7) to (336,) to match ISO requirements
        return features.flatten()

    def _get_obs(self):
        """
        Build observation vector using masked_adaptive encoding:
        - Zero-pad predictions after episode end
        - Include validity mask showing which predictions are valid
        - Include battery capacity constraints (available charge/discharge)
        """
        # Base PCS observations
        current_storage = self.pcs.get_total_storage() / 100.0
        current_consumption = self.pcs.get_consumption() / 10.0
        current_price = self._get_current_price() if self.settlement_mode == 'legacy' else None
        if self.settlement_mode == 'corrected':
            if self.current_step >= self.max_steps and self._settled_prices is not None:
                prices = self._settled_prices  # terminal observation has no next action
            else:
                prices = (self._get_current_price('buy'), self._get_current_price('sell'))
        else:
            prices = (current_price,)

        # Get raw predictions and normalize
        predicted_consumption = self._get_predicted_consumption(self.prediction_horizon) / 10.0
        
        # Apply masked_adaptive encoding
        steps_remaining = self.max_steps - self.current_step
        num_valid = min(len(predicted_consumption), steps_remaining)
        
        # Zero-pad predictions after episode end
        zero_padded = predicted_consumption.copy()
        zero_padded[num_valid:] = 0.0
        
        # Create validity mask (1 for valid within episode, 0 for beyond episode)
        mask = np.zeros(len(predicted_consumption))
        mask[:num_valid] = 1.0
        
        # Get battery capacity constraints
        battery_entity = self.pcs.storage_units[0] if len(self.pcs.storage_units) > 0 else None
        
        if battery_entity:
            try:
                # Get available capacities (normalized by same scale as actions)
                available_charge = float(battery_entity.get_available_charge_capacity()) / 10.0
                available_discharge = float(battery_entity.get_available_discharge_capacity()) / 10.0
            except Exception:
                # Fallback if methods don't exist
                available_charge = (100.0 - current_storage * 100.0) / 10.0  # space left
                available_discharge = (current_storage * 100.0) / 10.0  # current storage
        else:
            available_charge = 0.0
            available_discharge = 0.0
        
        # Get current time features
        current_features = self._get_current_features()

        # Concatenate: [storage, consumption, price] + [zero-padded] + [mask] + [capacities] + [features]
        obs = np.concatenate([
            [current_storage, current_consumption, *prices],
            zero_padded,
            mask,
            [available_charge, available_discharge],
            current_features
        ])

        obs = obs.astype(np.float32)
        if self.data_mode == 'corrected' and not np.isfinite(obs).all():
            raise ValueError("Observation contains non-finite consumption, price or features")
        return obs

    def render(self, action=None, verbosity=None):
        """
        Renders the current state of the environment with configurable verbosity.
        
        Args:
            action: Optional action array that will be applied. If provided, displays the action
                   that is about to be applied. If None, shows self.last_action (for backward compatibility).
            verbosity: Override instance verbosity level. If None, uses self.verbosity.
                Level 0: Silent - return data dict only, no output
                Level 1: Summary - episode totals only
                Level 2: Condensed - current step with key metrics (default)
                Level 3: Detailed - all state information for current step
                Level 4: Debug - includes battery dynamics and internal state
        
        Returns:
            dict: Structured data containing current state information
        """
        # Determine which verbosity level to use
        v = verbosity if verbosity is not None else self.verbosity
        
        # Determine which action to display
        if action is not None:
            display_action = float(action[0]) * self.action_scale
        else:
            display_action = self.last_action
        
        # Gather current state data
        current_price = self._get_current_price()
        strategy_name = self.price_strategy.__class__.__name__
        current_storage = self.pcs.get_total_storage()
        current_consumption = self.pcs.get_consumption()
        current_features = self._get_current_features()
        
        # Get battery details for higher verbosity levels
        battery_entity = self.pcs.storage_units[0] if len(self.pcs.storage_units) > 0 else None
        
        # Build structured data dict (returned at all verbosity levels)
        data = {
            'step': self.current_step,
            'max_steps': self.max_steps,
            'datetime': self.current_datetime,
            'storage': current_storage,
            'consumption': current_consumption,
            'price': current_price,
            'strategy': strategy_name,
            'last_action': self.last_action,
            'total_money': self.total_money_earned,
            'total_penalty': self.total_shortage_penalty,
            'total_reward': self.total_reward,
            'shortage_count': self.shortage_count,
        }
        
        # Add features if available
        if self.num_features > 0:
            data['features'] = {col: float(current_features[i]) for i, col in enumerate(self.feature_columns)}
        
        # Level 0: Silent - return data only
        if v == 0:
            return data
        
        # Level 1: Summary only (episode totals)
        if v == 1:
            if self.current_step == 0:  # Only print at episode start
                print(f"\n[PCS Episode Start] Date: {self.current_datetime.strftime('%Y-%m-%d')}")
            elif self.current_step >= self.max_steps - 1:  # Print at episode end
                print(f"\n{'=' * 60}")
                print(f"[PCS Episode Summary] {self.current_datetime.strftime('%Y-%m-%d')}")
                print(f"{'=' * 60}")
                print(f"Total Steps:           {self.current_step + 1}")
                print(f"Money Earned:          ${self.total_money_earned:.2f}")
                print(f"Shortage Penalties:    ${self.total_shortage_penalty:.2f}")
                print(f"Net Reward:            ${self.total_reward:.2f}")
                print(f"Shortage Count:        {self.shortage_count}")
                print(f"Final Storage:         {current_storage:.2f} units")
                print(f"Strategy Used:         {strategy_name}")
                print(f"{'=' * 60}\n")
            return data
        
        # Level 2: Condensed (current step with key metrics) - DEFAULT
        if v == 2:
            print(f"\n{'=' * 60}")
            print(f"[PCS Step {self.current_step}/{self.max_steps}] {self.current_datetime.strftime('%Y-%m-%d %H:%M')}")
            print(f"{'=' * 60}")
            print(f"Storage:               {current_storage:.2f} / 100.0 units")
            print(f"Consumption:           {current_consumption:.2f} units/step")
            print(f"Price ({strategy_name}): ${current_price:.4f}/unit")
            if display_action is not None:
                print(f"Battery Action:        {display_action:.2f} units")
            print(f"--- Episode Running Totals ---")
            print(f"Money:     ${self.total_money_earned:.2f}")
            print(f"Penalties: ${self.total_shortage_penalty:.2f}")
            print(f"Net:       ${self.total_reward:.2f}")
            print(f"Shortages: {self.shortage_count}")
            print(f"{'=' * 60}")
            return data
        
        # Level 3: Detailed (all state information)
        if v == 3:
            print(f"\n{'=' * 60}")
            print(f"[PCS DETAILED] Step {self.current_step}/{self.max_steps}")
            print(f"{'=' * 60}")
            print(f"Date/Time:             {self.current_datetime.strftime('%Y-%m-%d %H:%M')}")
            print(f"Storage Level:         {current_storage:.2f} / 100.0 units ({current_storage:.1f}%)")
            print(f"Consumption (Actual):  {current_consumption:.2f} units/step")
            print(f"Price ({strategy_name}): ${current_price:.4f}/unit")
            
            if self.last_action is not None:
                action_type = "Charge" if self.last_action < 0 else "Discharge" if self.last_action > 0 else "Idle"
                print(f"Battery Action:        {self.last_action:.2f} units ({action_type})")
            
            # Show features if available
            if self.num_features > 0:
                print(f"\n--- Environmental Features ---")
                for i, col in enumerate(self.feature_columns):
                    print(f"  {col:<20}: {current_features[i]:.4f}")
            
            print(f"\n--- Episode Cumulative Metrics ---")
            print(f"Gross Money Earned:    ${self.total_money_earned:.2f}")
            print(f"Shortage Penalties:    ${self.total_shortage_penalty:.2f}")
            print(f"Net Reward:            ${self.total_reward:.2f}")
            print(f"Shortage Occurrences:  {self.shortage_count}")
            print(f"{'=' * 60}")
            return data
        
        # Level 4: Debug (includes battery dynamics and internal state)
        if v >= 4:
            print(f"\n{'=' * 80}")
            print(f"[PCS DEBUG MODE] Step {self.current_step}/{self.max_steps}")
            print(f"{'=' * 80}")
            print(f"Date/Time:             {self.current_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Relative Time:         {self.current_step * self.dt:.4f} days")
            print(f"\n--- Storage System ---")
            print(f"Storage Level:         {current_storage:.4f} / 100.0 units ({current_storage:.2f}%)")
            
            if battery_entity:
                try:
                    avail_charge = float(battery_entity.get_available_charge_capacity())
                    avail_discharge = float(battery_entity.get_available_discharge_capacity())
                    data['available_charge'] = avail_charge
                    data['available_discharge'] = avail_discharge
                    print(f"Available Charge Cap:  {avail_charge:.4f} units")
                    print(f"Available Discharge:   {avail_discharge:.4f} units")
                except Exception as e:
                    print(f"Battery capacity info unavailable: {e}")
            
            print(f"\n--- Consumption & Pricing ---")
            print(f"Consumption (Actual):  {current_consumption:.4f} units/step")
            print(f"Pricing Strategy:      {strategy_name}")
            print(f"Current Price:         ${current_price:.6f}/unit")
            
            if self.last_action is not None:
                action_type = "CHARGE" if self.last_action < 0 else "DISCHARGE" if self.last_action > 0 else "IDLE"
                print(f"\n--- Agent Action ---")
                print(f"Raw Action Value:      {self.last_action:.4f} units")
                print(f"Action Type:           {action_type}")
                print(f"Action Bounds:         [-10.0, 10.0]")
            
            # Show all features in debug mode
            if self.num_features > 0:
                print(f"\n--- Environmental Features (All {self.num_features}) ---")
                for i, col in enumerate(self.feature_columns):
                    print(f"  {col:<25}: {current_features[i]:.6f}")
            
            # Show prediction horizon
            try:
                predictions = self._get_predicted_consumption(min(5, self.prediction_horizon))
                print(f"\n--- Consumption Predictions (Next 5 steps) ---")
                for i, pred in enumerate(predictions):
                    print(f"  Step +{i+1}: {pred:.4f} units")
            except Exception as e:
                print(f"\nPrediction data unavailable: {e}")
            
            print(f"\n--- Financial Metrics (Cumulative) ---")
            print(f"Gross Money Earned:    ${self.total_money_earned:.6f}")
            print(f"Shortage Penalties:    ${self.total_shortage_penalty:.6f}")
            print(f"Net Reward (Total):    ${self.total_reward:.6f}")
            print(f"Shortage Count:        {self.shortage_count}")
            if self.current_step > 0:
                avg_reward = self.total_reward / (self.current_step + 1)
                print(f"Average Reward/Step:   ${avg_reward:.6f}")
            
            print(f"{'=' * 80}")
            return data
        
        return data
    def _get_data_index(self, timestamp, allow_end=False):
        """Return an exact sample index, allowing the terminal timestamp when requested."""
        timestamp = pd.Timestamp(timestamp)
        if pd.isna(timestamp):
            raise ValueError("Selected date must be a finite consumption timestamp")
        offset = timestamp - self.test_start_date
        interval = pd.Timedelta(days=self.dt)
        index, remainder = divmod(offset.value, interval.value)
        upper_bound = len(self.test_df) if allow_end else len(self.test_df) - 1
        if remainder or index < 0 or index > upper_bound:
            raise ValueError("Selected date is outside consumption data or off the timestep grid")
        return int(index)

    def get_step_index(self) -> int:
        """
        Return the global half-hour step index since test_start_date.
        """
        if self.data_mode == 'corrected':
            return self._get_data_index(self.current_datetime, allow_end=True)
        days_from_start = (self.current_datetime - self.test_start_date).total_seconds() / 86400
        return int(days_from_start / self.dt)

    def set_step_index(self, idx: int):
        """
        Set the environment's current_datetime and intra-day current_step
        from a global half-hour index.
        """
        if self.data_mode == 'corrected':
            if not isinstance(idx, (int, np.integer)) or idx < 0 or idx >= len(self.test_df):
                raise ValueError("Step index must select a consumption sample")
            self.current_datetime = self.test_df.index[idx]
            self.current_step = int(idx % self.max_steps)
            return
        idx = int(idx)
        self.current_datetime = self.test_start_date + timedelta(minutes=30 * idx)
        self.current_step = int(idx % self.max_steps)

if __name__ == "__main__":
    # 1. Initialize the Environment
    # By default, PCSEnv will now use the SineWavePriceStrategy if no strategy is passed.
    # Using verbosity=2 (condensed) to show key metrics per step.
    env = PCSEnv(verbosity=2)

    # 2. Simulation Configuration
    num_days_to_run = 3
    accumulated_reward = 0.0
    render_every_n_steps = 1  # How often to call env.render() during the day

    # 3. Main Simulation Loop
    for day in range(num_days_to_run):
        # Reset at the start of each day.
        # This clears financial metrics and ensures the strategy starts fresh for the date.
        obs, info = env.reset()

        print(f"\n{'=' * 60}")
        print(f"STARTING DAY {day + 1}  |  Date: {env.current_datetime.strftime('%Y-%m-%d')}")
        print(f"{'=' * 60}")

        day_reward = 0.0

        # 4. Intra-day Time Steps (Typically 48 steps per day)
        while True:
            # Generate a random battery action (Intent) between -10 and 10.
            # -10: Maximum Charge, 10: Maximum Discharge.
            action_value = float(np.random.uniform(-10.0, 10.0))
            action = np.array([action_value], dtype=np.float32)

            # Render BEFORE stepping to show current state + action that will be applied
            if env.current_step % render_every_n_steps == 0:
                env.render(action=action)

            # Step the simulation.
            # The env will internally query the PriceCurveStrategy for the current price,
            # calculate the financial reward, and update the battery/consumption state.
            obs, reward, terminated, truncated, info = env.step(action)

            # Accumulate rewards (Net financial performance).
            day_reward += reward
            accumulated_reward += reward

            print(f"Step Reward: {reward:.2f}")

            # 5. Check for End of Day
            # Terminated becomes True once the current_step reaches max_steps (48).
            if terminated or truncated:
                print(f"\n>>> Day {day + 1} Finished!")
                print(f">>> Total Day Reward: {day_reward:.2f}")
                print(f">>> Final Storage: {info['storage_after_units']:.2f} units")
                print(f">>> Total Shortages Today: {env.shortage_count}")
                break

    # 6. Final Summary
    print(f"\n{'!' * 60}")
    print(f"Simulation Complete. Total Reward over {num_days_to_run} days: {accumulated_reward:.2f}")
    print(f"{'!' * 60}")
