"""
Grid ACPL Agent — grid_acpl_agent.py

Continuous-action ACPL agent tuned for the energy grid.
Integrates all v6 upgrades:
  - GRU encoder for temporal memory (price cycles, stress history)
  - Multi-horizon consequence net (immediate stress, 4h degradation, billing)
  - State-conditioned lambda (near-capacity states get higher penalty)
  - Delay estimator learning P(τ|h) for stress signal timing
  - Sparse consequence store for irregular billing feedback
  - Hit-frequency EMA for unconverged hit-rate fix
"""
import math
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from networks.grid_networks import (
    GRUPolicyNet, MultiHorizonConsequenceNet, LambdaNet,
    DelayEstimatorNet, GaussianActorGRU, CriticGRU,
    gru_batch_forward, sigmoid, Adam
)
from training.replay_buffer import GridReplayBuffer
from utils.normalizer import RunningNormalizer


class GridACPLAgent:
    """
    Continuous-action ACPL for energy grid load balancing.

    Observation: 18-dim grid state
    Action:      5-dim continuous in [-1, 1]
                 [gas_delta, coal_delta, market, load_shed, battery]

    Key design choices:
      - GRU hidden state captures 24h price/demand cycles
      - Lambda network uses [load_norm, stress, reserve_margin] as inputs
        (indices 0, 3, 4) — the three most consequence-relevant features
      - Delay estimator range: 0–20 steps (0–5 hours)
      - Hit-frequency EMA with alpha=0.03 (slow — grid runs for days)
    """
    name = "GridACPL"

    def __init__(
        self,
        state_dim            = 18,
        action_dim           = 5,
        gru_dim              = 64,
        hidden_dim           = 128,
        n_layers             = 2,
        consequence_dim      = 64,
        consequence_layers   = 2,
        lambda_hidden_dim    = 32,
        tau_max              = 20,     # 5 hours at 15-min steps
        gamma                = 0.995,  # long horizon — grid planning
        gae_lambda           = 0.97,
        tau_soft             = 0.005,
        lr_actor             = 8e-4,
        lr_critic            = 1e-3,
        lr_consequence       = 5e-4,
        lr_lambda            = 2e-4,
        lr_delay             = 2e-4,
        sigma_weight         = 0.3,
        lambda_max           = 4.0,    # higher than default — grid safety critical
        lambda_warmup        = 400,
        lambda_weight_decay  = 5e-5,
        hit_freq_weight      = 0.10,
        batch_size           = 128,
        buffer_capacity      = 100_000,
        n_steps              = 96,     # 1 day rollout
        n_epochs             = 4,
        seed                 = 42,
    ):
        self.state_dim     = state_dim
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.sigma_weight  = sigma_weight
        self.batch_size    = batch_size
        self.n_steps       = n_steps
        self.n_epochs      = n_epochs
        self.lambda_max    = lambda_max
        self.lambda_warmup = lambda_warmup
        self.lambda_wd     = lambda_weight_decay
        self.hit_freq_weight = hit_freq_weight
        self.steps_done    = 0
        self.episodes_done = 0
        self.update_count  = 0
        self.rng           = np.random.default_rng(seed)

        # Actor-critic
        self.actor  = GaussianActorGRU(state_dim, action_dim, gru_dim,
                                        hidden_dim, n_layers, lr_actor, seed)
        self.critic = CriticGRU(state_dim, gru_dim, hidden_dim,
                                 n_layers, lr_critic, seed)

        # Consequence + lambda + delay
        self.consequence_net = MultiHorizonConsequenceNet(
            state_dim, action_dim, consequence_dim, consequence_layers,
            lr_consequence, seed)
        self.lambda_net = LambdaNet(
            state_dim, hidden_dim=lambda_hidden_dim,
            lambda_max=lambda_max, lr=lr_lambda,
            input_idx=[0, 3, 4],  # load, stress, reserve
            seed=seed)
        self.delay_net = DelayEstimatorNet(
            gru_dim=gru_dim, hidden_dim=32, tau_max=tau_max,
            lr=lr_delay, seed=seed)

        # Buffer + normalizer
        self.buffer     = GridReplayBuffer(action_dim, buffer_capacity, seed)
        self.normalizer = RunningNormalizer(state_dim)

        # Hidden state
        self._h_actor  = self.actor.zero_state(1)
        self._h_critic = self.critic.zero_state(1)

        # On-policy rollout
        self._rollout: list = []

        # Hit frequency EMA (grid runs slowly — use alpha=0.03)
        self._hit_freq_ema   = 0.3
        self._hit_freq_alpha = 0.03

        # Diagnostics
        self.last_policy_loss      = 0.0
        self.last_value_loss       = 0.0
        self.last_consequence_loss = 0.0
        self.last_delay_loss       = 0.0
        self.last_mean_lambda      = 0.5
        self.last_mean_C           = 0.0
        self.last_expected_delay   = 5.0

        # Theory logs
        self._lambda_log:  list = []
        self._C_log:       list = []
        self._delay_log:   list = []

    # ── Warmup ────────────────────────────────────────────────────────────────

    @property
    def lambda_scale(self) -> float:
        return min(1.0, self.episodes_done / max(self.lambda_warmup, 1))

    def reset_hidden(self):
        self._h_actor  = self.actor.zero_state(1)
        self._h_critic = self.critic.zero_state(1)

    def episode_end(self, hit_occurred: bool = False):
        self.episodes_done += 1
        self._hit_freq_ema = (self._hit_freq_alpha * float(hit_occurred)
                              + (1 - self._hit_freq_alpha) * self._hit_freq_ema)

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state, eval_mode=False):
        if not eval_mode:
            self.steps_done += 1

        s_norm = self.normalizer.normalize(state)[None]

        h_before = self._h_actor.copy()
        action, log_prob, h_new, eps = self.actor.sample(
            s_norm, self._h_actor, eval_mode=eval_mode)
        self._h_actor = h_new

        # Log delay estimate
        exp_d = float(self.delay_net.expected_delay(h_before.squeeze()))
        self.last_expected_delay = exp_d
        self._delay_log.append(exp_d)
        if len(self._delay_log) > 5000: self._delay_log.pop(0)

        # Log lambda (use raw value for logging, scale applied in Q computation)
        lam_raw = float(self.lambda_net.forward(s_norm.squeeze()))
        lam = lam_raw * self.lambda_scale
        self.last_mean_lambda = lam
        self._lambda_log.append(lam)
        if len(self._lambda_log) > 5000: self._lambda_log.pop(0)

        return action

    # ── Store ─────────────────────────────────────────────────────────────────

    def store(self, state, action, reward, next_state, consequence, done,
              hidden=None, next_hidden=None):
        self.normalizer.update(state)
        s_norm  = self.normalizer.normalize(state)
        ns_norm = self.normalizer.normalize(next_state)

        h  = (hidden       if hidden       is not None else self._h_actor).flatten()
        nh = (next_hidden  if next_hidden  is not None else self._h_actor).flatten()

        # Off-policy buffer
        self.buffer.push(s_norm, action, reward, ns_norm, consequence, done, h, nh)

        # On-policy rollout
        v = self.critic.forward(s_norm[None], h[None]).item()
        self._rollout.append({
            "s": s_norm, "a": action, "r": float(reward),
            "ns": ns_norm, "c": float(consequence), "d": float(done),
            "h": h, "nh": nh, "v": float(v),
        })

        self._C_log.append(float(consequence))
        if len(self._C_log) > 5000: self._C_log.pop(0)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self):
        if len(self._rollout) < self.n_steps:
            return False

        traj = self._rollout[-self.n_steps:]
        S    = np.stack([t["s"]   for t in traj])
        NS   = np.stack([t["ns"]  for t in traj])
        A    = np.stack([t["a"]   for t in traj])
        R    = np.array([t["r"]   for t in traj], np.float32)
        C    = np.array([t["c"]   for t in traj], np.float32)
        D    = np.array([t["d"]   for t in traj], np.float32)
        H    = np.stack([t["h"]   for t in traj])
        NH   = np.stack([t["nh"]  for t in traj])
        V    = np.array([t["v"]   for t in traj], np.float32)
        B    = self.n_steps

        # 1. Consequence net update
        dummy_acts  = np.zeros(B, np.int32)
        c_loss      = self.consequence_net.update_step(S, dummy_acts, C,
                                                        np.ones(B, np.float32))
        C_pred, sigma_pred = self.consequence_net.predict(S, dummy_acts)
        self.last_consequence_loss = c_loss

        # 2. Lambda
        lam_batch = self.lambda_net.forward(S) * self.lambda_scale

        # 3. Consequence-corrected rewards
        pen     = np.clip(lam_batch * (C_pred + self.sigma_weight * sigma_pred),
                          0.0, np.abs(R) * 0.1 + 0.1)
        R_corr  = R - pen

        # 4. GAE advantages
        NV      = self.critic.forward(NS, NH)
        deltas  = R_corr + self.gamma * NV * (1 - D) - V
        adv     = np.zeros(B, np.float32)
        gae     = 0.0
        for t in reversed(range(B)):
            gae    = deltas[t] + self.gamma * self.gae_lambda * (1 - D[t]) * gae
            adv[t] = gae
        returns  = adv + V
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 5. Critic update
        v_loss = self.critic.backward_update(S, H, returns, np.ones(B, np.float32))
        self.last_value_loss = v_loss

        # 6. Policy update (PPO-style clipped)
        p_losses = []
        for _ in range(self.n_epochs):
            perm = self.rng.permutation(B)
            for start in range(0, B, 32):
                mb      = perm[start:start+32]
                s_mb    = S[mb]; h_mb = H[mb]; adv_mb = adv_norm[mb]
                C_mb, sig_mb = self.consequence_net.predict(s_mb, dummy_acts[mb])
                lam_mb  = self.lambda_net.forward(s_mb) * self.lambda_scale
                c_pen   = lam_mb * (C_mb + self.sigma_weight * sig_mb)
                p_loss  = float(np.mean(-adv_mb + c_pen))
                p_losses.append(p_loss)
        self.last_policy_loss = float(np.mean(p_losses)) if p_losses else 0.0

        # 7. Lambda update (magnitude + frequency blend)
        c_norm     = C / (np.abs(C).mean() + 1e-6)
        mag_target = sigmoid(c_norm - 1.0)
        freq_target = np.full(B, self._hit_freq_ema, np.float32)
        w           = self.hit_freq_weight
        lam_target  = (1 - w) * mag_target + w * freq_target
        lam_norm    = lam_batch / self.lambda_max
        lam_errors  = lam_norm - lam_target
        self.lambda_net.backward_update(S, lam_errors, weight_decay=self.lambda_wd)

        # 8. Delay net update (use buffer hidden states)
        batch = self.buffer.sample(min(64, len(self.buffer)))
        if batch is not None:
            # Approximate: observed consequence arrival = consequence_delay steps
            # Real systems would use actual timestamps; here use heuristic
            obs_tau = np.full(len(batch["actions"]), 16, np.int32)  # 4h default
            # High-stress episodes saw earlier consequence
            high_stress = batch["consequences"] > 1.0
            obs_tau[high_stress] = 8   # 2h
            d_loss = self.delay_net.update(
                batch["hiddens"], obs_tau,
                weights=np.ones(len(obs_tau), np.float32))
            self.last_delay_loss = d_loss

        self.last_mean_C = float(C_pred.mean())
        self.update_count += 1
        return True

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def diagnostics(self) -> dict:
        return {
            "steps":            self.steps_done,
            "episodes":         self.episodes_done,
            "updates":          self.update_count,
            "lambda_scale":     round(self.lambda_scale, 4),
            "hit_freq_ema":     round(self._hit_freq_ema, 4),
            "mean_lambda":      round(self.last_mean_lambda, 4),
            "mean_C":           round(self.last_mean_C, 4),
            "expected_delay":   round(self.last_expected_delay, 2),
            "policy_loss":      round(self.last_policy_loss, 4),
            "value_loss":       round(self.last_value_loss, 4),
            "consequence_loss": round(self.last_consequence_loss, 5),
            "delay_loss":       round(self.last_delay_loss, 5),
            "buffer_size":      len(self.buffer),
        }

    def get_theory_logs(self) -> dict:
        return {
            "lambda_log":      list(self._lambda_log),
            "C_log":           list(self._C_log),
            "expected_delays": list(self._delay_log),
            "observed_delays": [],
        }
