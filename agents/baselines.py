"""agents/baselines.py — Rule-based and random baseline agents for comparison"""
import numpy as np


class RandomAgent:
    """Purely random continuous actions — lower bound baseline."""
    name = "RandomAgent"
    def __init__(self, action_dim=5, seed=42):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
    def select_action(self, state, eval_mode=False):
        return self.rng.uniform(-1, 1, self.action_dim).astype(np.float32)
    def store(self, *a, **kw): pass
    def update(self): return False
    def reset_hidden(self): pass
    def episode_end(self, **kw): pass
    def diagnostics(self): return {}


class RuleBasedAgent:
    """
    Expert rule-based controller — upper-bound heuristic.
    Rules:
      1. If load > 85% capacity → ramp gas, buy spot, shed if critical
      2. If renewable high → reduce gas/coal
      3. Charge battery when spot cheap + renewable surplus
      4. Discharge battery at peak demand
      5. Keep coal flat (slow ramp)
      6. Never stress nuclear (nuclear barely changes)
    """
    name = "RuleBase"

    def __init__(self, action_dim=5, seed=42):
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)

    def select_action(self, state, eval_mode=False):
        # State indices:
        # 0=load_norm, 1=renew_frac, 2=spot_price_norm, 3=stress, 4=reserve
        # 5=freq_dev, 6=gas_norm, 7=coal_norm, 8=nuc_norm, 9=renew_norm
        # 10=batt_soc, 11-12=hour_sin/cos, 13-14=dow_sin/cos
        # 15=temp_norm, 16=stress_24h, 17=market_trend

        load         = float(state[0])
        spot_norm    = float(state[2])
        stress       = float(state[3])
        reserve      = float(state[4])
        freq_dev     = float(state[5])
        gas_norm     = float(state[6])
        batt_soc     = float(state[10])
        hour_sin     = float(state[11])
        hour_cos     = float(state[12])

        # Approximate hour from sin/cos
        hour = (np.arctan2(hour_sin, hour_cos) / (2*np.pi) * 24) % 24
        is_peak = (7 < hour < 10) or (18 < hour < 22)
        is_cheap = spot_norm < 0.3

        # 1. Gas delta: ramp up if overloaded, ramp down if reserve large
        if load > 0.85 or freq_dev < -0.3:
            gas_delta = min(1.0, (load - 0.8) * 4)
        elif reserve > 0.20:
            gas_delta = -0.3
        else:
            gas_delta = 0.0

        # 2. Coal: keep steady (slow ramp)
        coal_delta = 0.0

        # 3. Market: buy when cheap + needed, sell when expensive + surplus
        if is_cheap and batt_soc < 0.7:
            market = 0.5
        elif spot_norm > 0.7 and reserve > 0.10:
            market = -0.4   # sell to spot
        elif load > 0.90:
            market = 0.8    # emergency purchase
        else:
            market = 0.0

        # 4. Load shed: only in emergency
        if load > 0.95 and reserve < 0.02:
            load_shed = 0.8
        elif load > 0.90 and freq_dev < -0.4:
            load_shed = 0.4
        else:
            load_shed = -0.1  # slowly restore

        # 5. Battery: charge off-peak cheap, discharge peak
        if is_peak and batt_soc > 0.3:
            battery = -0.7   # discharge
        elif is_cheap and batt_soc < 0.85:
            battery = 0.6    # charge
        elif load > 0.88 and batt_soc > 0.2:
            battery = -0.5   # emergency discharge
        else:
            battery = 0.0

        action = np.array([gas_delta, coal_delta, market, load_shed, battery], np.float32)
        # Add small noise to avoid being too deterministic
        action += self.rng.normal(0, 0.05, self.action_dim).astype(np.float32)
        return np.clip(action, -1.0, 1.0)

    def store(self, *a, **kw): pass
    def update(self): return False
    def reset_hidden(self): pass
    def episode_end(self, **kw): pass
    def diagnostics(self): return {}
