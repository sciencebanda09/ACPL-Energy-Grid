"""
Energy Grid Environment — grid_env.py

Models a regional power grid with:
  - 4 generator types: gas, coal, nuclear, renewable
  - Spot market buying/selling
  - Load shedding capability
  - Equipment stress / degradation
  - Delayed consequences: billing (monthly), degradation (4h), stability (instant)

State space (18-dim):
  [0]  grid_load_norm          current demand / max_capacity
  [1]  renewable_fraction      fraction of supply from renewables
  [2]  spot_price_norm         current spot market price (normalised)
  [3]  equipment_stress        cumulative stress 0-1
  [4]  reserve_margin          headroom above demand (0-1)
  [5]  frequency_deviation     grid Hz deviation from 50Hz (normalised)
  [6]  gas_output_norm         gas generator output / capacity
  [7]  coal_output_norm        coal generator output / capacity
  [8]  nuclear_output_norm     nuclear output / capacity
  [9]  renewable_output_norm   renewable output / capacity
  [10] battery_soc             battery state of charge 0-1
  [11] hour_of_day_sin         time encoding sin
  [12] hour_of_day_cos         time encoding cos
  [13] day_of_week_sin
  [14] day_of_week_cos
  [15] temperature_norm        outside temp (affects demand)
  [16] cumulative_stress_24h   rolling 24h stress index
  [17] market_trend            spot price momentum

Action space (5-dim continuous, each in [-1, 1]):
  [0] gas_delta        ramp gas generator up (+) or down (-)
  [1] coal_delta       ramp coal
  [2] market_action    buy from spot (+) or sell to spot (-)
  [3] load_shed        shed load (+ = shed, - = restore)
  [4] battery_action   charge battery (+) or discharge (-)

Consequence signal (delayed):
  - Immediate: frequency deviation penalty (grid stability)
  - 4h delay:  equipment stress signal (overload → degradation)
  - 24h delay: efficiency cost (running suboptimal mix)
  - 30d delay: billing/penalty from grid operator (constraint violations)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List


# ── Grid constants ─────────────────────────────────────────────────────────────

MAX_CAPACITY_MW   = 1000.0   # total installed capacity
NOMINAL_FREQ      = 50.0     # Hz
FREQ_DEADBAND     = 0.1      # Hz — no penalty inside band
FREQ_EMERGENCY    = 0.5      # Hz — emergency threshold

GENERATOR_CAPS = {           # MW per generator type
    "gas":       350.0,
    "coal":      300.0,
    "nuclear":   200.0,
    "renewable": 150.0,      # stochastic (wind+solar)
}
GENERATOR_RAMP = {           # max MW change per timestep (15 min)
    "gas":       80.0,
    "coal":      30.0,
    "nuclear":   5.0,        # nuclear barely ramps
    "renewable": 0.0,        # uncontrollable
}
GENERATOR_COST = {           # $/MWh variable cost
    "gas":       65.0,
    "coal":      45.0,
    "nuclear":   20.0,
    "renewable": 5.0,
}
GENERATOR_STRESS = {         # stress coefficient when running >80% capacity
    "gas":       0.3,
    "coal":      0.5,
    "nuclear":   0.8,        # nuclear stress is serious
    "renewable": 0.0,
}

BATTERY_CAPACITY_MWH = 200.0
BATTERY_POWER_MW     = 100.0  # max charge/discharge rate

LOAD_SHED_COST       = 500.0  # $/MWh — customer penalty
SPOT_PRICE_BASE      = 80.0   # $/MWh
DEGRADATION_COST     = 2000.0 # $/stress-unit

# Timestep = 15 minutes; 4h = 16 steps; 24h = 96 steps
TIMESTEP_HOURS       = 0.25
STRESS_DELAY_STEPS   = 16     # 4 hours
EFFICIENCY_DELAY     = 96     # 24 hours
BILLING_DELAY        = 2880   # 30 days × 96 steps/day


@dataclass
class GridState:
    gas_mw:          float = 200.0
    coal_mw:         float = 150.0
    nuclear_mw:      float = 180.0
    renewable_mw:    float = 80.0
    battery_soc:     float = 0.5
    load_shed_mw:    float = 0.0
    spot_bought_mw:  float = 0.0
    equipment_stress: float = 0.0
    cumulative_stress_24h: float = 0.0
    frequency:       float = 50.0
    hour:            int   = 0
    day:             int   = 0
    spot_price:      float = 80.0
    temperature:     float = 20.0
    market_trend:    float = 0.0


class EnergyGridEnv:
    """
    Energy grid load balancing environment.
    Implements the BaseEnv interface for ACPL compatibility.
    """
    name       = "energy_grid"
    state_dim  = 18
    action_dim = 5
    constraint_threshold = 15.0  # max acceptable consequence per episode

    def __init__(self, max_steps: int = 96,       # 1 day of 15-min steps
                 consequence_delay: int = 16,      # 4 hours
                 noise_std: float = 0.03,
                 seed: int = 0,
                 difficulty: str = "normal"):      # easy / normal / hard
        self.max_steps          = max_steps
        self.consequence_delay  = consequence_delay
        self.noise_std          = noise_std
        self.difficulty         = difficulty
        self.rng                = np.random.default_rng(seed)

        self._step              = 0
        self._done              = False
        self._total_reward      = 0.0
        self._total_consequence = 0.0
        self._delayed_hits      = 0
        self._consequence_queue: List[float] = []
        self._stress_history:   List[float] = []  # rolling 24h
        self._billing_acc:      float = 0.0       # 30-day accumulator

        self.gs = GridState()
        self._demand_profile = self._build_demand_profile()

    # ── Demand profile ─────────────────────────────────────────────────────────

    def _build_demand_profile(self) -> np.ndarray:
        """Synthetic 24h demand curve (MW), 15-min resolution."""
        hours  = np.linspace(0, 24, 96, endpoint=False)
        # Morning peak ~8am, evening peak ~7pm, night trough ~3am
        base   = 550.0
        morn   = 120 * np.exp(-0.5 * ((hours - 8) / 1.5)**2)
        eve    = 150 * np.exp(-0.5 * ((hours - 19) / 1.5)**2)
        night  = -80 * np.exp(-0.5 * ((hours - 3)  / 2.0)**2)
        return base + morn + eve + night

    def _current_demand(self) -> float:
        h = self._step % 96
        base = self._demand_profile[h]
        # Temperature effect on demand
        temp_effect = 8.0 * max(0, self.gs.temperature - 22) \
                    + 6.0 * max(0, 15 - self.gs.temperature)
        noise = self.rng.normal(0, self.noise_std * base)
        diff  = {"easy": 0.85, "normal": 1.0, "hard": 1.15}[self.difficulty]
        return max(100.0, base * diff + temp_effect + noise)

    def _renewable_output(self) -> float:
        """Stochastic renewable output based on hour + weather."""
        h = self._step % 96
        hour = h * 0.25
        solar = max(0, GENERATOR_CAPS["renewable"] * 0.7
                    * np.sin(np.pi * (hour - 6) / 12)
                    * (hour > 6 and hour < 18))
        wind  = GENERATOR_CAPS["renewable"] * 0.3 \
                * (0.5 + 0.5 * np.sin(self._step * 0.1))
        noise = self.rng.normal(0, 10.0)
        return float(np.clip(solar + wind + noise, 0, GENERATOR_CAPS["renewable"]))

    def _spot_price(self) -> float:
        """Dynamic spot price with demand correlation + volatility."""
        h     = self._step % 96
        hour  = h * 0.25
        base  = SPOT_PRICE_BASE
        peak  = 40 * (np.exp(-0.5*((hour-8)/2)**2) + np.exp(-0.5*((hour-19)/2)**2))
        vol   = self.rng.normal(0, 12.0)
        trend = self.gs.market_trend * 5.0
        return float(np.clip(base + peak + vol + trend, 20.0, 400.0))

    def _temperature(self) -> float:
        h    = self._step % 96
        hour = h * 0.25
        base = 15 + 8 * np.sin(np.pi * (hour - 6) / 12)
        return float(base + self.rng.normal(0, 2.0))

    # ── State encoding ─────────────────────────────────────────────────────────

    def _encode_state(self) -> np.ndarray:
        gs  = self.gs
        h   = self._step % 96
        hour = h * 0.25
        dow  = (self._step // 96) % 7

        total_supply = (gs.gas_mw + gs.coal_mw + gs.nuclear_mw
                        + gs.renewable_mw + gs.spot_bought_mw
                        + max(0, -gs.battery_soc * BATTERY_POWER_MW)
                        - gs.load_shed_mw)
        demand       = self._current_demand()
        reserve      = max(0, total_supply - demand) / MAX_CAPACITY_MW

        s = np.array([
            demand / MAX_CAPACITY_MW,
            gs.renewable_mw / (total_supply + 1e-3),
            gs.spot_price / (SPOT_PRICE_BASE * 3),
            gs.equipment_stress,
            reserve,
            (gs.frequency - NOMINAL_FREQ) / FREQ_EMERGENCY,
            gs.gas_mw       / GENERATOR_CAPS["gas"],
            gs.coal_mw      / GENERATOR_CAPS["coal"],
            gs.nuclear_mw   / GENERATOR_CAPS["nuclear"],
            gs.renewable_mw / GENERATOR_CAPS["renewable"],
            gs.battery_soc,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow  / 7),
            np.cos(2 * np.pi * dow  / 7),
            (gs.temperature - 15) / 20,
            gs.cumulative_stress_24h,
            gs.market_trend / 3.0,
        ], dtype=np.float32)
        return np.clip(s, -2.0, 2.0)

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        start_hour = self.rng.integers(0, 96)
        self._step              = int(start_hour)
        self._done              = False
        self._total_reward      = 0.0
        self._total_consequence = 0.0
        self._delayed_hits      = 0
        self._consequence_queue = []
        self._stress_history    = []
        self._billing_acc       = 0.0

        self.gs = GridState(
            gas_mw        = float(self.rng.uniform(150, 280)),
            coal_mw       = float(self.rng.uniform(100, 220)),
            nuclear_mw    = float(self.rng.uniform(160, 200)),
            renewable_mw  = self._renewable_output(),
            battery_soc   = float(self.rng.uniform(0.3, 0.7)),
            hour          = int(start_hour),
            spot_price    = self._spot_price(),
            temperature   = self._temperature(),
        )
        return self._encode_state()

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, float, bool, dict]:
        assert not self._done
        action = np.asarray(action, np.float32)
        action = np.clip(action, -1.0, 1.0)

        gs = self.gs

        # 1. Decode actions
        gas_delta   = float(action[0]) * GENERATOR_RAMP["gas"]
        coal_delta  = float(action[1]) * GENERATOR_RAMP["coal"]
        mkt_action  = float(action[2]) * 80.0   # ±80 MW from spot
        shed_action = float(action[3])           # +1 = shed, -1 = restore
        batt_action = float(action[4]) * BATTERY_POWER_MW

        # 2. Apply generator ramps
        gs.gas_mw  = float(np.clip(gs.gas_mw  + gas_delta,
                                   0, GENERATOR_CAPS["gas"]))
        gs.coal_mw = float(np.clip(gs.coal_mw + coal_delta,
                                   0, GENERATOR_CAPS["coal"]))
        # Nuclear barely ramps — small noise only
        gs.nuclear_mw = float(np.clip(
            gs.nuclear_mw + self.rng.normal(0, 2.0), 170, 200))

        # 3. Renewable (uncontrollable)
        gs.renewable_mw = self._renewable_output()

        # 4. Spot market
        gs.spot_bought_mw = float(np.clip(mkt_action, -50, 80))

        # 5. Battery
        batt_delta = float(np.clip(batt_action,
                                   -BATTERY_POWER_MW, BATTERY_POWER_MW))
        energy_kwh  = batt_delta * TIMESTEP_HOURS
        new_soc     = gs.battery_soc + energy_kwh / BATTERY_CAPACITY_MWH
        gs.battery_soc = float(np.clip(new_soc, 0.05, 0.95))
        batt_mw_net = -batt_delta   # negative = discharging into grid

        # 6. Load shedding
        max_shed    = self._current_demand() * 0.20  # max 20% can be shed
        if shed_action > 0:
            shed_delta = shed_action * max_shed * 0.1
            gs.load_shed_mw = float(np.clip(gs.load_shed_mw + shed_delta, 0, max_shed))
        else:
            restore = abs(shed_action) * gs.load_shed_mw * 0.2
            gs.load_shed_mw = float(max(0, gs.load_shed_mw - restore))

        # 7. Supply / demand balance
        demand  = self._current_demand() - gs.load_shed_mw
        supply  = (gs.gas_mw + gs.coal_mw + gs.nuclear_mw
                   + gs.renewable_mw + gs.spot_bought_mw + batt_mw_net)
        imbalance_mw = supply - demand

        # 8. Frequency deviation (proportional to imbalance)
        freq_dev    = imbalance_mw / MAX_CAPACITY_MW * 2.0   # Hz
        gs.frequency = float(np.clip(NOMINAL_FREQ + freq_dev,
                                     NOMINAL_FREQ - 2, NOMINAL_FREQ + 2))

        # 9. Equipment stress
        stress_now = 0.0
        for gen, cap, s_coef in [
            ("gas",     GENERATOR_CAPS["gas"],     GENERATOR_STRESS["gas"]),
            ("coal",    GENERATOR_CAPS["coal"],    GENERATOR_STRESS["coal"]),
            ("nuclear", GENERATOR_CAPS["nuclear"], GENERATOR_STRESS["nuclear"]),
        ]:
            mw    = getattr(gs, f"{gen}_mw")
            ratio = mw / cap
            if ratio > 0.85:
                stress_now += s_coef * (ratio - 0.85) / 0.15

        gs.equipment_stress = float(np.clip(
            gs.equipment_stress * 0.99 + stress_now * 0.01, 0, 1))

        # Rolling 24h stress
        self._stress_history.append(stress_now)
        if len(self._stress_history) > 96:
            self._stress_history.pop(0)
        gs.cumulative_stress_24h = float(
            np.mean(self._stress_history) if self._stress_history else 0)

        # 10. Update market
        gs.spot_price   = self._spot_price()
        gs.temperature  = self._temperature()
        gs.market_trend = float(0.9 * gs.market_trend
                                + 0.1 * self.rng.normal(0, 1.0))

        # ── Reward computation ────────────────────────────────────────────────
        # Operating cost (negative = cost → negative reward)
        gen_cost = (
            gs.gas_mw   * GENERATOR_COST["gas"]   * TIMESTEP_HOURS +
            gs.coal_mw  * GENERATOR_COST["coal"]  * TIMESTEP_HOURS +
            gs.nuclear_mw * GENERATOR_COST["nuclear"] * TIMESTEP_HOURS +
            gs.renewable_mw * GENERATOR_COST["renewable"] * TIMESTEP_HOURS
        )
        mkt_cost   = gs.spot_bought_mw * gs.spot_price * TIMESTEP_HOURS
        shed_cost  = gs.load_shed_mw * LOAD_SHED_COST * TIMESTEP_HOURS
        freq_pen   = 50.0 * max(0, abs(gs.frequency - NOMINAL_FREQ) - FREQ_DEADBAND)
        total_cost = gen_cost + mkt_cost + shed_cost + freq_pen

        # Normalise reward to [-1, +1] range
        reward = float(-(total_cost / 5000.0))
        # Bonus for renewable utilisation
        renewable_bonus = float(gs.renewable_mw / GENERATOR_CAPS["renewable"] * 0.2)
        # Bonus for being within reserve margin
        reserve_bonus   = 0.1 if 0.05 < (supply - demand) / MAX_CAPACITY_MW < 0.15 else 0.0
        reward += renewable_bonus + reserve_bonus

        # ── Consequence computation (delayed) ─────────────────────────────────
        # Immediate consequence: frequency instability
        freq_c = float(max(0, abs(gs.frequency - NOMINAL_FREQ) - FREQ_DEADBAND) * 2.0)

        # Delayed consequence: stress signal (pushed to queue)
        stress_c = float(stress_now * 3.0)

        # Billing consequence (accumulates, measured periodically)
        self._billing_acc += shed_cost * 0.001
        billing_c = 0.0
        if self._step % 96 == 95:   # end of day
            billing_c = float(self._billing_acc * 0.5)
            self._billing_acc = 0.0

        immediate_c = freq_c + billing_c

        # Push stress_c into delay queue (arrives in consequence_delay steps)
        self._consequence_queue.append(stress_c)
        if len(self._consequence_queue) > self.consequence_delay:
            delayed_c = self._consequence_queue.pop(0)
        else:
            delayed_c = 0.0

        total_consequence = immediate_c + delayed_c

        if total_consequence > 1.0:
            self._delayed_hits += 1

        self._total_reward      += reward
        self._total_consequence += total_consequence
        self._step              += 1
        self._done = (self._step >= self.max_steps + (self._step - int(self._step % self.max_steps or self.max_steps)))

        # Simpler done check:
        steps_taken = self._step - (self._step % self.max_steps or self.max_steps) if False else None
        self._done  = (self._step % self.max_steps == 0 and self._step > 0)

        next_state = self._encode_state()

        info = {
            "demand_mw":      demand + gs.load_shed_mw,
            "supply_mw":      supply,
            "imbalance_mw":   imbalance_mw,
            "frequency":      gs.frequency,
            "equipment_stress": gs.equipment_stress,
            "spot_price":     gs.spot_price,
            "load_shed_mw":   gs.load_shed_mw,
            "battery_soc":    gs.battery_soc,
            "gen_cost":       gen_cost,
            "immediate_c":    immediate_c,
            "delayed_c":      delayed_c,
            "stress_now":     stress_now,
            "renewable_mw":   gs.renewable_mw,
            "resource_load":  demand / MAX_CAPACITY_MW,  # ACPL compat
            "delayed_hits":   self._delayed_hits,
        }
        return next_state, reward, total_consequence, self._done, info

    @property
    def done(self): return self._done

    def episode_stats(self) -> dict:
        return {
            "total_reward":      self._total_reward,
            "total_consequence": self._total_consequence,
            "delayed_hits":      self._delayed_hits,
            "steps":             self._step,
            "csr_violation":     int(self._total_consequence > self.constraint_threshold),
        }


# ── Difficulty variants ────────────────────────────────────────────────────────

class EasyGridEnv(EnergyGridEnv):
    name = "grid_easy"
    def __init__(self, **kw): super().__init__(difficulty="easy", **kw)

class HardGridEnv(EnergyGridEnv):
    name = "grid_hard"
    def __init__(self, **kw): super().__init__(difficulty="hard", **kw)

class StormGridEnv(EnergyGridEnv):
    """Extreme renewable variability — stress tests delay estimation."""
    name = "grid_storm"
    def __init__(self, **kw):
        super().__init__(difficulty="hard", noise_std=0.15, **kw)

    def _renewable_output(self) -> float:
        # Intermittent — can drop to zero or spike suddenly
        base = super()._renewable_output()
        if self.rng.random() < 0.05:   # 5% chance of sudden cloud/calm
            return float(self.rng.uniform(0, 20))
        return base

class PeakDemandGridEnv(EnergyGridEnv):
    """Heatwave scenario — sustained high demand."""
    name = "grid_peak"
    def __init__(self, **kw): super().__init__(difficulty="hard", **kw)

    def _current_demand(self) -> float:
        return super()._current_demand() * 1.25  # 25% higher everywhere


# ── Registry (matches ACPL ENV_REGISTRY pattern) ──────────────────────────────

GRID_ENV_REGISTRY = {
    "energy_grid": EnergyGridEnv,
    "grid_easy":   EasyGridEnv,
    "grid_hard":   HardGridEnv,
    "grid_storm":  StormGridEnv,
    "grid_peak":   PeakDemandGridEnv,
}

GRID_TRAIN_ENVS  = ("energy_grid", "grid_easy", "grid_hard")
GRID_EVAL_ENVS   = ("energy_grid", "grid_hard")
GRID_UNSEEN_ENVS = ("grid_storm", "grid_peak")
