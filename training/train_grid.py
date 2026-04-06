"""
training/train_grid.py — Training loop for energy grid agents
"""
import os, json, time, datetime
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from environments.grid_env import GRID_ENV_REGISTRY, GRID_TRAIN_ENVS, EnergyGridEnv


def _fmt_eta(sec):
    if sec < 0: return "?"
    h, r = divmod(int(sec), 3600); m, s = divmod(r, 60)
    if h: return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def _clean(v):
    if isinstance(v, (float, int, bool, str)): return v
    if hasattr(v, 'item'): return v.item()
    if isinstance(v, (list, tuple)): return [_clean(x) for x in v]
    if isinstance(v, dict): return {k: _clean(vv) for k, vv in v.items()}
    return float(v) if hasattr(v, '__float__') else str(v)


def run_episode(agent, env, train=True, update_freq=4):
    """Run one episode. Returns result dict."""
    agent.reset_hidden()
    state = env.reset()
    ep_r = ep_c = 0.0
    ep_steps = 0
    losses, infer_times = [], []
    hit_occurred = False
    is_acpl = hasattr(agent, "_h_actor")
    h_before = agent._h_actor.copy() if is_acpl else None
    info_last = {}

    while not env.done:
        t0     = time.perf_counter()
        action = agent.select_action(state, eval_mode=not train)
        infer_times.append(time.perf_counter() - t0)

        h_after = agent._h_actor.copy() if is_acpl else None
        ns, r, c, done, info = env.step(action)
        info_last = info

        if info.get("delayed_hits", 0) > 0:
            hit_occurred = True

        if train:
            if is_acpl:
                agent.store(state, action, r, ns, c, done, h_before, h_after)
            else:
                agent.store(state, action, r, ns, c, done)
            if ep_steps % update_freq == 0:
                if agent.update():
                    if hasattr(agent, "last_policy_loss"):
                        losses.append(agent.last_policy_loss)

        h_before = h_after
        state    = ns
        ep_r    += r
        ep_c    += c
        ep_steps += 1

    stats = env.episode_stats()
    if train and hasattr(agent, "episode_end"):
        agent.episode_end(hit_occurred=hit_occurred)

    return {
        "episode_reward":      ep_r,
        "episode_consequence": ep_c,
        "delayed_hits":        stats["delayed_hits"],
        "steps":               ep_steps,
        "mean_loss":           float(np.mean(losses)) if losses else 0.0,
        "mean_infer_ms":       float(np.mean(infer_times)*1000) if infer_times else 0.0,
        "hit_occurred":        hit_occurred,
        "frequency":           info_last.get("frequency", 50.0),
        "equipment_stress":    info_last.get("equipment_stress", 0.0),
        "load_shed_mw":        info_last.get("load_shed_mw", 0.0),
        "battery_soc":         info_last.get("battery_soc", 0.5),
    }


def train_agent(agent, n_episodes=500, max_steps=96, delay_steps=16,
                update_freq=4, log_freq=25, seed=42, verbose=True,
                env_names=GRID_TRAIN_ENVS, log_dir=None, save_every=50):

    env_list = [GRID_ENV_REGISTRY[n] for n in env_names]
    rng      = np.random.default_rng(seed)

    history = {k: [] for k in (
        "rewards", "consequences", "delayed_hits", "losses",
        "cumulative_reward", "cumulative_consequence",
        "env_name", "episode_time_s", "infer_ms",
        "frequency", "equipment_stress", "load_shed_mw", "battery_soc",
        "mean_lambda", "hit_freq_ema", "lambda_scale", "expected_delay")}

    cum_r = cum_c = 0.0
    ep_times = []
    train_start = time.time()

    for ep in range(1, n_episodes + 1):
        EnvCls = env_list[rng.integers(len(env_list))]
        env    = EnvCls(max_steps=max_steps, consequence_delay=delay_steps, seed=seed+ep)

        ep_start = time.perf_counter()
        result   = run_episode(agent, env, train=True, update_freq=update_freq)
        ep_time  = time.perf_counter() - ep_start
        ep_times.append(ep_time)

        cum_r += result["episode_reward"]
        cum_c += result["episode_consequence"]

        history["rewards"].append(result["episode_reward"])
        history["consequences"].append(result["episode_consequence"])
        history["delayed_hits"].append(result["delayed_hits"])
        history["losses"].append(result["mean_loss"])
        history["cumulative_reward"].append(cum_r)
        history["cumulative_consequence"].append(cum_c)
        history["env_name"].append(env.name)
        history["episode_time_s"].append(ep_time)
        history["infer_ms"].append(result["mean_infer_ms"])
        history["frequency"].append(result["frequency"])
        history["equipment_stress"].append(result["equipment_stress"])
        history["load_shed_mw"].append(result["load_shed_mw"])
        history["battery_soc"].append(result["battery_soc"])

        diag = agent.diagnostics() if hasattr(agent, "diagnostics") else {}
        history["mean_lambda"].append(diag.get("mean_lambda", 0.0))
        history["hit_freq_ema"].append(diag.get("hit_freq_ema", 0.0))
        history["lambda_scale"].append(diag.get("lambda_scale", 1.0))
        history["expected_delay"].append(diag.get("expected_delay", 0.0))

        if log_dir and ep % save_every == 0:
            _save_history(history, agent, log_dir)

        if verbose and ep % log_freq == 0:
            avg_t   = float(np.mean(ep_times[-30:])) if ep_times else 0
            eta     = _fmt_eta((n_episodes - ep) * avg_t)
            elapsed = _fmt_eta(time.time() - train_start)
            diag    = agent.diagnostics() if hasattr(agent, "diagnostics") else {}
            print(
                f"[{agent.name}] Ep {ep:4d}/{n_episodes} | "
                f"Elapsed {elapsed} | ETA {eta} | "
                f"Env: {env.name:12s} | "
                f"R: {result['episode_reward']:6.2f} | "
                f"C: {result['episode_consequence']:5.3f} | "
                f"Hits: {result['delayed_hits']:2d} | "
                f"λ: {diag.get('mean_lambda',0):.3f} | "
                f"λ_w: {diag.get('lambda_scale',1):.2f} | "
                f"hit%: {diag.get('hit_freq_ema',0):.2f} | "
                f"E[τ]: {diag.get('expected_delay',0):.1f}steps | "
                f"Freq: {result['frequency']:.2f}Hz | "
                f"Stress: {result['equipment_stress']:.3f}"
            )

    history["total_train_s"] = time.time() - train_start
    if log_dir:
        path = _save_history(history, agent, log_dir)
        if verbose: print(f"  [log] Training history → {path}")
    return history


def _save_history(history, agent, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    name = getattr(agent, "name", "agent").replace(" ", "_")
    path = os.path.join(log_dir, f"{name}_training.json")
    with open(path, "w") as f:
        json.dump(_clean(history), f, indent=2)
    return path


def evaluate_agent(agent, env_name, n_episodes=50, max_steps=96,
                   delay_steps=16, seed_offset=9999):
    EnvClass = GRID_ENV_REGISTRY[env_name]
    rewards, consequences, delayed_hits = [], [], []
    frequencies, stresses, infer_times  = [], [], []
    csr_violations = []
    threshold = getattr(EnvClass, "constraint_threshold", 15.0)

    for ep in range(n_episodes):
        env = EnvClass(max_steps=max_steps, consequence_delay=delay_steps,
                       seed=seed_offset + ep)
        result = run_episode(agent, env, train=False)
        rewards.append(result["episode_reward"])
        consequences.append(result["episode_consequence"])
        delayed_hits.append(result["delayed_hits"])
        frequencies.append(result["frequency"])
        stresses.append(result["equipment_stress"])
        infer_times.append(result["mean_infer_ms"])
        csr_violations.append(int(result["episode_consequence"] > threshold))

    csr = 100.0 * (1.0 - float(np.mean(csr_violations)))
    return {
        "mean_reward":       float(np.mean(rewards)),
        "std_reward":        float(np.std(rewards)),
        "mean_consequence":  float(np.mean(consequences)),
        "std_consequence":   float(np.std(consequences)),
        "mean_delayed_hits": float(np.mean(delayed_hits)),
        "mean_frequency":    float(np.mean(frequencies)),
        "mean_stress":       float(np.mean(stresses)),
        "mean_infer_ms":     float(np.mean(infer_times)),
        "csr":               csr,
        "stability":         float(1.0/(np.std(rewards)+1.0)),
        "rewards":           rewards,
        "consequences":      consequences,
    }


def evaluate_all(agents, env_names, n_episodes=50, max_steps=96,
                 delay_steps=16, log_dir=None):
    results = {}
    for name, agent in agents.items():
        results[name] = {}
        for env_name in env_names:
            print(f"  Evaluating {name} on {env_name}...")
            results[name][env_name] = evaluate_agent(
                agent, env_name, n_episodes, max_steps, delay_steps)

    if log_dir:
        path = os.path.join(log_dir, "eval_results.json")
        with open(path, "w") as f:
            json.dump(_clean(results), f, indent=2)
        print(f"  [log] Eval results → {path}")
    return results


def print_results_table(results, env_names):
    agents = list(results.keys())
    print("\n" + "="*100)
    print(f"{'ENERGY GRID BENCHMARK':^100}")
    print("="*100)
    hdr = f"{'Agent':<18} | {'Reward':>8} | {'J_c':>8} | {'CSR%':>6} | {'Hits':>6} | {'Freq(Hz)':>8} | {'Stress':>7} | {'ms/step':>7}"
    print(hdr); print("-"*100)
    for ag in agents:
        all_r = [results[ag][e]["mean_reward"]      for e in env_names]
        all_c = [results[ag][e]["mean_consequence"] for e in env_names]
        all_csr=[results[ag][e]["csr"]              for e in env_names]
        all_dh= [results[ag][e]["mean_delayed_hits"]for e in env_names]
        all_f = [results[ag][e]["mean_frequency"]   for e in env_names]
        all_s = [results[ag][e]["mean_stress"]      for e in env_names]
        all_ms= [results[ag][e]["mean_infer_ms"]    for e in env_names]
        print(f"{ag:<18} | {np.mean(all_r):>8.3f} | {np.mean(all_c):>8.4f} | "
              f"{np.mean(all_csr):>5.1f}% | {np.mean(all_dh):>6.2f} | "
              f"{np.mean(all_f):>8.3f} | {np.mean(all_s):>7.4f} | {np.mean(all_ms):>7.3f}")
    print("="*100)
