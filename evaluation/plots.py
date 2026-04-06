"""evaluation/plots.py — Grid evaluation plots"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


COLORS = {
    "GridACPL":   "#2196F3",
    "RuleBase":   "#FF9800",
    "RandomAgent":"#9E9E9E",
    "MPC":        "#4CAF50",
    "default":    "#607D8B",
}

def _color(name):
    for k, v in COLORS.items():
        if k in name: return v
    return COLORS["default"]


def smooth(arr, w=10):
    if len(arr) < w: return np.array(arr)
    kernel = np.ones(w)/w
    return np.convolve(arr, kernel, mode="valid")


def plot_training_curves(histories: dict, out_dir: str):
    """Plot reward, consequence, frequency, stress, lambda over training."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Energy Grid — Training Curves", fontsize=14, fontweight="bold")

    metrics = [
        ("rewards",          "Episode Reward",          axes[0,0], True),
        ("consequences",     "Consequence (J_c)",        axes[0,1], True),
        ("frequency",        "Grid Frequency (Hz)",      axes[1,0], False),
        ("equipment_stress", "Equipment Stress",         axes[1,1], True),
        ("mean_lambda",      "Mean λ(s)",                axes[2,0], False),
        ("hit_freq_ema",     "Hit Frequency EMA",        axes[2,1], False),
    ]

    for key, title, ax, do_smooth in metrics:
        for name, hist in histories.items():
            vals = hist.get(key, [])
            if not vals: continue
            arr = np.array(vals)
            x   = np.arange(len(arr))
            ax.plot(x, arr, alpha=0.2, color=_color(name))
            if do_smooth and len(arr) > 10:
                s = smooth(arr, 20)
                ax.plot(np.arange(len(s)), s, label=name, color=_color(name), linewidth=2)
            else:
                ax.plot(x, arr, label=name, color=_color(name), linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Episode")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if key == "frequency":
            ax.axhline(50.0, color="red", linestyle="--", alpha=0.5, label="Nominal 50Hz")
            ax.axhline(49.5, color="orange", linestyle=":", alpha=0.5)
            ax.axhline(50.5, color="orange", linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "01_training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_grid_operations(env_history: dict, out_dir: str, episode: int = 0):
    """Plot a single episode's grid operations timeline."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Grid Operations — Episode {episode}", fontsize=13, fontweight="bold")

    steps = np.arange(len(env_history.get("demand", [])))

    def _plot(ax, keys_labels_colors, title, ylabel, hlines=None):
        for key, label, color in keys_labels_colors:
            vals = env_history.get(key, [])
            if vals: ax.plot(steps[:len(vals)], vals, label=label, color=color, linewidth=1.5)
        if hlines:
            for y, ls, c in hlines:
                ax.axhline(y, linestyle=ls, color=c, alpha=0.6)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="upper right"); ax.grid(True, alpha=0.3)

    _plot(axes[0],
          [("demand_mw","Demand","#F44336"),
           ("supply_mw","Supply","#2196F3"),
           ("load_shed_mw","Shed","#FF9800")],
          "Supply / Demand Balance", "MW")

    _plot(axes[1],
          [("gas_mw","Gas","#FF5722"),
           ("coal_mw","Coal","#795548"),
           ("nuclear_mw","Nuclear","#9C27B0"),
           ("renewable_mw","Renewable","#4CAF50"),
           ("spot_bought_mw","Spot","#00BCD4")],
          "Generator Dispatch", "MW")

    _plot(axes[2],
          [("frequency","Frequency","#2196F3")],
          "Grid Frequency", "Hz",
          hlines=[(50.0,"--","red"),(49.5,":","orange"),(50.5,":","orange")])

    _plot(axes[3],
          [("equipment_stress","Stress","#F44336"),
           ("battery_soc","Battery SoC","#4CAF50"),
           ("lambda_val","λ(s)","#9C27B0")],
          "Equipment Stress / Battery / Lambda", "0–1")

    axes[3].set_xlabel("Timestep (15-min intervals)")
    plt.tight_layout()
    path = os.path.join(out_dir, f"02_grid_operations_ep{episode}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_lambda_heatmap(state_lambda_pairs: list, out_dir: str):
    """
    Visualise lambda values across the state space.
    state_lambda_pairs: list of (load_norm, stress, lambda_val)
    """
    if not state_lambda_pairs: return
    arr = np.array(state_lambda_pairs)
    load  = arr[:, 0]; stress = arr[:, 1]; lam = arr[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("State-Conditioned Lambda λ(s) — Risk Map", fontsize=13, fontweight="bold")

    # Scatter: load vs lambda
    sc = axes[0].scatter(load, lam, c=stress, cmap="YlOrRd", alpha=0.5, s=10)
    axes[0].set_xlabel("Grid Load (normalised)")
    axes[0].set_ylabel("λ(s) — Consequence Weight")
    axes[0].set_title("λ vs Grid Load (colour = stress)")
    plt.colorbar(sc, ax=axes[0], label="Equipment Stress")
    axes[0].axvline(0.85, color="red", linestyle="--", alpha=0.7, label="85% capacity")
    axes[0].legend()

    # Heatmap: load × stress → lambda
    lb = np.linspace(0, 1, 20); sb = np.linspace(0, 1, 20)
    grid_lam = np.zeros((20, 20))
    for i, l in enumerate(lb):
        for j, s in enumerate(sb):
            mask = (np.abs(load - l) < 0.05) & (np.abs(stress - s) < 0.05)
            grid_lam[j, i] = lam[mask].mean() if mask.sum() > 0 else np.nan
    im = axes[1].imshow(grid_lam, origin="lower", aspect="auto",
                         extent=[0,1,0,1], cmap="YlOrRd", vmin=0)
    axes[1].set_xlabel("Grid Load"); axes[1].set_ylabel("Equipment Stress")
    axes[1].set_title("λ(load, stress) heatmap")
    plt.colorbar(im, ax=axes[1], label="λ(s)")

    plt.tight_layout()
    path = os.path.join(out_dir, "03_lambda_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_delay_distribution(delay_history: list, out_dir: str):
    """Plot the learned delay distribution P(τ|h) evolution."""
    if not delay_history: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Delay Estimator — P(τ|h) Analysis", fontsize=13, fontweight="bold")

    axes[0].hist(delay_history, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
    axes[0].axvline(16, color="red", linestyle="--", label="True delay (4h=16 steps)")
    axes[0].set_xlabel("Expected Delay E[τ|h] (steps)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Expected Delay Estimates")
    axes[0].legend()

    # Evolution over time
    n   = len(delay_history)
    x   = np.arange(n)
    s   = smooth(np.array(delay_history), 20)
    axes[1].plot(x[:len(s)], s, color="#2196F3", linewidth=2)
    axes[1].axhline(16, color="red", linestyle="--", alpha=0.7, label="True 4h delay")
    axes[1].fill_between(x[:len(s)], s-2, s+2, alpha=0.2, color="#2196F3")
    axes[1].set_xlabel("Training Step"); axes[1].set_ylabel("E[τ|h] (steps)")
    axes[1].set_title("Delay Estimate Convergence")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "04_delay_estimator.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_benchmark_comparison(results: dict, env_names: list, out_dir: str):
    """Bar chart comparison across agents."""
    agents  = list(results.keys())
    metrics = ["mean_reward", "mean_consequence", "csr", "mean_stress"]
    titles  = ["Mean Reward ↑", "Mean J_c ↓", "CSR% ↑", "Equipment Stress ↓"]
    bests   = ["max", "min", "max", "min"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Benchmark Comparison — All Environments", fontsize=13, fontweight="bold")

    for ax, metric, title, best in zip(axes, metrics, titles, bests):
        vals = [np.mean([results[ag][e][metric] for e in env_names]) for ag in agents]
        colors = [_color(ag) for ag in agents]
        bars = ax.bar(agents, vals, color=colors, alpha=0.85, edgecolor="white")
        # Annotate best
        best_val = max(vals) if best == "max" else min(vals)
        for bar, v in zip(bars, vals):
            if abs(v - best_val) < 1e-6:
                bar.set_edgecolor("gold"); bar.set_linewidth(2.5)
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01*max(abs(v) for v in vals if v!=0 or True),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(title, fontsize=10); ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30); ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "05_benchmark_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")
    return path


def generate_all_plots(histories, results, env_names, out_dir,
                       state_lambda_pairs=None, delay_history=None):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    if histories: paths.append(plot_training_curves(histories, out_dir))
    if results:   paths.append(plot_benchmark_comparison(results, env_names, out_dir))
    if state_lambda_pairs: paths.append(plot_lambda_heatmap(state_lambda_pairs, out_dir))
    if delay_history:      paths.append(plot_delay_distribution(delay_history, out_dir))
    return [p for p in paths if p]
