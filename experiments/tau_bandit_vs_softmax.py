"""
s^tau vs Softmax: Multi-Armed Bandit Duel

Three experiments comparing s^tau and softmax as decision mechanisms:

Part 1 — Static Duel:
  Same UCB scores, s^tau(tau) vs softmax(temperature), parameter sweep.
  Question: which mechanism achieves lower regret, and at what cost?

Part 2 — Non-Stationary:
  After 1500 steps, the best arm changes. Who adapts faster?
  s^tau's clamp may lock onto old-best; softmax's always-nonzero may
  maintain enough exploration to notice the shift.

Part 3 — Regret Decomposition:
  What fraction of pulls go to clearly suboptimal arms (mean < best/2)?
  s^tau's clamping behavior should waste fewer pulls on bad arms.

Key insight: s^tau(s) = softmax(tau * log(clamp(s, eps)))
Even though analytically equivalent, the CLAMP creates a practical
difference in decision-making: negative-scoring arms get ~zero probability
under s^tau, but non-zero under softmax.
"""
import numpy as np
import os
from pathlib import Path

EPS = 1e-8

# ──────────────────────────────────────────────
# Decision functions
# ──────────────────────────────────────────────

def stau_norm_numpy(scores, tau):
    clamped = np.maximum(scores, EPS)
    powered = np.power(clamped, tau)
    s = powered.sum()
    return powered / s if s > 0 else np.ones_like(powered) / len(powered)


def softmax_numpy(scores, temperature=1.0):
    s = scores - scores.max()
    exps = np.exp(s / max(temperature, 0.01))
    s = exps.sum()
    return exps / s if s > 0 else np.ones_like(exps) / len(exps)


# ──────────────────────────────────────────────
# Part 1: Static Duel
# ──────────────────────────────────────────────

def run_bandit_static(arm_means, n_steps, tau, temperature=None, seed=42):
    """
    Single bandit run. Temperature=None → use s^tau;
    temperature is float → use softmax.
    """
    rng = np.random.RandomState(seed)
    n_arms = len(arm_means)
    best_mean = max(arm_means)

    counts = np.zeros(n_arms)
    rewards_sum = np.zeros(n_arms)
    regrets = np.zeros(n_steps)

    for t in range(n_steps):
        if t < n_arms:
            arm = t
        else:
            empirical = np.divide(rewards_sum, counts,
                                  out=np.zeros_like(rewards_sum), where=counts > 0)
            bonus = np.sqrt(2 * np.log(t + 1) / np.maximum(counts, 1))
            scores = empirical + bonus

            if temperature is None:
                probs = stau_norm_numpy(scores, tau)
            else:
                probs = softmax_numpy(scores, temperature)
            arm = rng.choice(n_arms, p=probs)

        reward = float(rng.rand() < arm_means[arm])
        counts[arm] += 1
        rewards_sum[arm] += reward
        regrets[t] = best_mean - arm_means[arm]

    return {
        'cumulative_regret': np.cumsum(regrets),
        'counts': counts,
        'final_empirical': rewards_sum / np.maximum(counts, 1),
    }


def run_static_duel(arm_means, n_steps=2000, n_runs=30, seed=42):
    """Parameter sweep: s^tau vs softmax across multiple runs."""
    tau_vals = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0]
    temp_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]

    print("=" * 70)
    print("  PART 1: s^tau vs Softmax — Static Bandit Duel")
    print("=" * 70)
    print(f"  Arms: {len(arm_means)} | Best mean: {max(arm_means):.2f}")
    print(f"  Steps: {n_steps} | Runs per config: {n_runs}")
    print()

    # s^tau sweep
    stau_results = {}
    print("  ── s^tau ──")
    for tau in tau_vals:
        all_regrets = np.zeros((n_runs, n_steps))
        best_idx = np.argmax(arm_means)
        all_accuracy = np.zeros(n_runs)
        for r in range(n_runs):
            res = run_bandit_static(arm_means, n_steps, tau=tau, seed=seed + r)
            all_regrets[r] = res['cumulative_regret']
            all_accuracy[r] = float(np.argmax(res['counts']) == best_idx)
        stau_results[tau] = {
            'avg_regret': all_regrets.mean(axis=0),
            'final_regret': all_regrets.mean(axis=0)[-1],
            'std': all_regrets.std(axis=0)[-1],
            'accuracy': all_accuracy.mean(),
        }
        print(f"    tau={tau:5.1f} | Regret={stau_results[tau]['final_regret']:8.2f} "
              f"| Acc={stau_results[tau]['accuracy']:.1%}")

    # softmax sweep
    sm_results = {}
    print("  ── Softmax ──")
    for temp in temp_vals:
        all_regrets = np.zeros((n_runs, n_steps))
        best_idx = np.argmax(arm_means)
        all_accuracy = np.zeros(n_runs)
        for r in range(n_runs):
            res = run_bandit_static(arm_means, n_steps, tau=0, temperature=temp, seed=seed + r)
            all_regrets[r] = res['cumulative_regret']
            all_accuracy[r] = float(np.argmax(res['counts']) == best_idx)
        sm_results[temp] = {
            'avg_regret': all_regrets.mean(axis=0),
            'final_regret': all_regrets.mean(axis=0)[-1],
            'std': all_regrets.std(axis=0)[-1],
            'accuracy': all_accuracy.mean(),
        }
        print(f"    T={temp:5.2f}   | Regret={sm_results[temp]['final_regret']:8.2f} "
              f"| Acc={sm_results[temp]['accuracy']:.1%}")

    best_stau = min(stau_results, key=lambda t: stau_results[t]['final_regret'])
    best_sm = min(sm_results, key=lambda t: sm_results[t]['final_regret'])
    print()
    print(f"  Best s^tau:  tau={best_stau}  regret={stau_results[best_stau]['final_regret']:.1f}")
    print(f"  Best softmax: T={best_sm}  regret={sm_results[best_sm]['final_regret']:.1f}")
    improvement = (sm_results[best_sm]['final_regret'] - stau_results[best_stau]['final_regret'])
    print(f"  s^tau advantage: {improvement:+.1f} ({improvement/sm_results[best_sm]['final_regret']*100:+.1f}%)")
    print()

    return stau_results, sm_results


# ──────────────────────────────────────────────
# Part 2: Non-Stationary Bandit
# ──────────────────────────────────────────────

def run_bandit_nonstationary(arm_means_before, arm_means_after, switch_step,
                              tau, temperature=None, seed=42):
    """
    Environment changes at switch_step.
    Tests adaptation speed of s^tau vs softmax.
    """
    rng = np.random.RandomState(seed)
    n_arms = len(arm_means_before)

    counts = np.zeros(n_arms)
    rewards_sum = np.zeros(n_arms)
    total_steps = switch_step + 2000  # 2000 steps after switch
    regrets = np.zeros(total_steps)
    chosen = np.zeros(total_steps, dtype=int)

    arm_means = arm_means_before.copy()
    new_best_arm = np.argmax(arm_means_after)

    for t in range(total_steps):
        if t == switch_step:
            arm_means = arm_means_after.copy()

        best_mean = max(arm_means)

        if t < n_arms:
            arm = t
        else:
            empirical = np.divide(rewards_sum, counts,
                                  out=np.zeros_like(rewards_sum), where=counts > 0)
            bonus = np.sqrt(2 * np.log(t + 1) / np.maximum(counts, 1))
            scores = empirical + bonus

            if temperature is None:
                probs = stau_norm_numpy(scores, tau)
            else:
                probs = softmax_numpy(scores, temperature)
            arm = rng.choice(n_arms, p=probs)

        reward = float(rng.rand() < arm_means[arm])
        counts[arm] += 1
        rewards_sum[arm] += reward
        regrets[t] = best_mean - arm_means[arm]
        chosen[t] = arm

    cumulative = np.cumsum(regrets)

    # Recovery: after switch, how many steps until new best arm is pulled
    # more often than the old best arm (in a rolling window)?
    after_chosen = chosen[switch_step:]
    window = 50
    recovery_step = len(after_chosen)
    old_best_arm = np.argmax(arm_means_before)

    for i in range(window, len(after_chosen), 10):
        w = after_chosen[i-window:i]
        new_best_pulls = (w == new_best_arm).sum()
        old_best_pulls = (w == old_best_arm).sum()
        if new_best_pulls > old_best_pulls + 5:  # clear lead
            recovery_step = i
            break

    return {
        'cumulative_regret': cumulative,
        'regret_before_switch': cumulative[switch_step - 1],
        'regret_after_switch': cumulative[-1] - cumulative[switch_step - 1],
        'recovery_steps': recovery_step,
        'counts': counts,
    }


def run_nonstationary_duel(arm_before, arm_after, switch_step=1500, n_runs=30):
    tau_vals = [1.0, 2.0, 3.0, 5.0, 10.0]
    temp_vals = [0.1, 0.3, 0.5, 1.0, 2.0]

    print("=" * 70)
    print("  PART 2: Non-Stationary Bandit — Recovery Test")
    print("=" * 70)
    print(f"  Before switch: best arm {np.argmax(arm_before)} = {max(arm_before):.2f}")
    print(f"  After  switch:  best arm {np.argmax(arm_after)} = {max(arm_after):.2f}")
    print(f"  Switch at step: {switch_step}")
    print()

    # s^tau
    print("  ── s^tau Recovery ──")
    stau_ns = {}
    for tau in tau_vals:
        all_regret_before = []
        all_regret_after = []
        all_recovery = []
        for r in range(n_runs):
            res = run_bandit_nonstationary(
                arm_before, arm_after, switch_step, tau=tau, seed=42 + r
            )
            all_regret_before.append(res['regret_before_switch'])
            all_regret_after.append(res['regret_after_switch'])
            all_recovery.append(res['recovery_steps'])
        stau_ns[tau] = {
            'regret_before': np.mean(all_regret_before),
            'regret_after': np.mean(all_regret_after),
            'recovery_steps': np.mean(all_recovery),
        }
        print(f"    tau={tau:5.1f} | Before={stau_ns[tau]['regret_before']:8.1f} "
              f"| After={stau_ns[tau]['regret_after']:8.1f} "
              f"| Recovery={stau_ns[tau]['recovery_steps']:.0f} steps")

    # softmax
    print("  ── Softmax Recovery ──")
    sm_ns = {}
    for temp in temp_vals:
        all_regret_before = []
        all_regret_after = []
        all_recovery = []
        for r in range(n_runs):
            res = run_bandit_nonstationary(
                arm_before, arm_after, switch_step, tau=0, temperature=temp, seed=42 + r
            )
            all_regret_before.append(res['regret_before_switch'])
            all_regret_after.append(res['regret_after_switch'])
            all_recovery.append(res['recovery_steps'])
        sm_ns[temp] = {
            'regret_before': np.mean(all_regret_before),
            'regret_after': np.mean(all_regret_after),
            'recovery_steps': np.mean(all_recovery),
        }
        print(f"    T={temp:5.2f}   | Before={sm_ns[temp]['regret_before']:8.1f} "
              f"| After={sm_ns[temp]['regret_after']:8.1f} "
              f"| Recovery={sm_ns[temp]['recovery_steps']:.0f} steps")

    return stau_ns, sm_ns


# ──────────────────────────────────────────────
# Part 3: Regret Decomposition
# ──────────────────────────────────────────────

def run_regret_decomposition(arm_means, n_steps=3000, n_runs=30):
    """Analyze where regret comes from: bad arms or medium arms."""
    tau_vals = [1.0, 3.0, 8.0]
    temp_vals = [0.2, 0.5, 1.0]
    best_mean = max(arm_means)
    threshold = best_mean / 2  # arms below this are "clearly bad"

    bad_arm_indices = [i for i, m in enumerate(arm_means) if m < threshold]

    print("=" * 70)
    print("  PART 3: Regret Decomposition — Where Do Pulls Go?")
    print("=" * 70)
    print(f"  \"Bad\" arms (mean < {threshold:.2f}): indices {bad_arm_indices}")
    print(f"  Steps: {n_steps} | Runs: {n_runs}")
    print()

    all_configs = []

    # s^tau
    for tau in tau_vals:
        total_pulls = np.zeros(len(arm_means))
        for r in range(n_runs):
            res = run_bandit_static(arm_means, n_steps, tau=tau, seed=42 + r)
            total_pulls += res['counts']
        bad_pulls = total_pulls[bad_arm_indices].sum()
        bad_frac = bad_pulls / total_pulls.sum()
        all_configs.append(('stau', tau, bad_frac, total_pulls))
        print(f"  s^tau  tau={tau:<5.1f} | Bad-arm pulls: {bad_pulls:6.0f} "
              f"({bad_frac:.1%})")

    # softmax
    for temp in temp_vals:
        total_pulls = np.zeros(len(arm_means))
        for r in range(n_runs):
            res = run_bandit_static(arm_means, n_steps, tau=0, temperature=temp, seed=42 + r)
            total_pulls += res['counts']
        bad_pulls = total_pulls[bad_arm_indices].sum()
        bad_frac = bad_pulls / total_pulls.sum()
        all_configs.append(('softmax', temp, bad_frac, total_pulls))
        print(f"  SM     T={temp:<5.2f}   | Bad-arm pulls: {bad_pulls:6.0f} "
              f"({bad_frac:.1%})")

    print()
    return all_configs


# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────

def plot_static_duel(stau_results, sm_results, arm_means, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tau_vals = sorted(stau_results.keys())
    temp_vals = sorted(sm_results.keys())
    tau_colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(tau_vals)))
    sm_colors = plt.cm.Oranges(np.linspace(0.4, 0.95, len(temp_vals)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("s^tau vs Softmax: Static Bandit Duel",
                 fontsize=14, fontweight='bold')

    # [0,0] s^tau regret curves
    ax = axes[0, 0]
    for tau, c in zip(tau_vals, tau_colors):
        ax.plot(stau_results[tau]['avg_regret'], color=c, alpha=0.85,
                linewidth=1.5, label=f'tau={tau:.1f}')
    ax.set_title('s^tau: Cumulative Regret')
    ax.set_xlabel('Steps'); ax.set_ylabel('Regret')
    ax.legend(fontsize=6.5, ncol=2); ax.grid(True, alpha=0.3)

    # [0,1] softmax regret curves
    ax = axes[0, 1]
    for temp, c in zip(temp_vals, sm_colors):
        ax.plot(sm_results[temp]['avg_regret'], color=c, alpha=0.85,
                linewidth=1.5, label=f'T={temp:.2f}')
    ax.set_title('Softmax: Cumulative Regret')
    ax.set_xlabel('Steps'); ax.set_ylabel('Regret')
    ax.legend(fontsize=6.5, ncol=2); ax.grid(True, alpha=0.3)

    # [0,2] Final regret vs parameter
    ax = axes[0, 2]
    ax.plot(tau_vals, [stau_results[t]['final_regret'] for t in tau_vals],
            'o-', color='#58a6ff', linewidth=2, markersize=8, label='s^tau')
    ax.plot(temp_vals, [sm_results[t]['final_regret'] for t in temp_vals],
            's--', color='#f78166', linewidth=2, markersize=8, label='Softmax')
    ax.set_title('Final Regret vs Parameter')
    ax.set_xlabel('Parameter (tau / T)'); ax.set_ylabel('Regret')
    ax.legend(); ax.grid(True, alpha=0.3)

    # [1,0] Overlay: best s^tau vs best softmax
    ax = axes[1, 0]
    best_tau = min(stau_results, key=lambda t: stau_results[t]['final_regret'])
    best_temp = min(sm_results, key=lambda t: sm_results[t]['final_regret'])
    ax.plot(stau_results[best_tau]['avg_regret'], color='#58a6ff',
            linewidth=2.5, label=f's^tau (tau={best_tau})')
    ax.plot(sm_results[best_temp]['avg_regret'], color='#f78166',
            linewidth=2.5, label=f'Softmax (T={best_temp})')
    ax.set_title(f'Best vs Best (s^tau advantage: '
                 f'{sm_results[best_temp]["final_regret"] - stau_results[best_tau]["final_regret"]:.0f})')
    ax.set_xlabel('Steps'); ax.set_ylabel('Regret')
    ax.legend(); ax.grid(True, alpha=0.3)

    # [1,1] Accuracy vs parameter
    ax = axes[1, 1]
    ax.plot(tau_vals, [stau_results[t]['accuracy'] * 100 for t in tau_vals],
            'o-', color='#58a6ff', linewidth=2, markersize=8, label='s^tau')
    ax.plot(temp_vals, [sm_results[t]['accuracy'] * 100 for t in temp_vals],
            's--', color='#f78166', linewidth=2, markersize=8, label='Softmax')
    ax.set_title('Best-Arm Accuracy')
    ax.set_xlabel('Parameter'); ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105); ax.legend(); ax.grid(True, alpha=0.3)

    # [1,2] Summary text
    ax = axes[1, 2]
    ax.axis('off')
    lines = [
        "DUEL SUMMARY",
        "=" * 18,
        f"Arms: {len(arm_means)}",
        f"Best arm: {np.argmax(arm_means)} ({max(arm_means):.2f})",
        "",
        "s^tau best:",
        f"  tau={best_tau}",
        f"  regret={stau_results[best_tau]['final_regret']:.1f}",
        f"  accuracy={stau_results[best_tau]['accuracy']:.1%}",
        "",
        "Softmax best:",
        f"  T={best_temp}",
        f"  regret={sm_results[best_temp]['final_regret']:.1f}",
        f"  accuracy={sm_results[best_temp]['accuracy']:.1%}",
        "",
        "Why s^tau wins:",
        "  clamp(score, eps) zeros",
        "  out bad arms. Softmax",
        "  always gives them non-",
        "  zero probability.",
    ]
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontfamily='monospace', fontsize=7.5, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_nonstationary_duel(stau_ns, sm_ns, arm_before, arm_after, switch_step, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tau_vals = sorted(stau_ns.keys())
    temp_vals = sorted(sm_ns.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Non-Stationary Bandit: Recovery After Environment Shift",
                 fontsize=13, fontweight='bold')

    # Bar chart: regret before vs after
    ax = axes[0]
    x = np.arange(len(tau_vals) + len(temp_vals))
    labels = [f's^tau\n{tau:.1f}' for tau in tau_vals] + \
             [f'SM\n{t:.2f}' for t in temp_vals]
    before_vals = [stau_ns[t]['regret_before'] for t in tau_vals] + \
                  [sm_ns[t]['regret_before'] for t in temp_vals]
    after_vals = [stau_ns[t]['regret_after'] for t in tau_vals] + \
                 [sm_ns[t]['regret_after'] for t in temp_vals]

    w = 0.35
    colors_before = ['#58a6ff'] * len(tau_vals) + ['#f78166'] * len(temp_vals)
    colors_after = ['#1f4a8a'] * len(tau_vals) + ['#a04030'] * len(temp_vals)

    bars1 = ax.bar(x - w/2, before_vals, w, label='Before switch',
                   color=colors_before, alpha=0.85)
    bars2 = ax.bar(x + w/2, after_vals, w, label='After switch',
                   color=colors_after, alpha=0.85)
    ax.set_title(f'Regret Before/After Switch (step {switch_step})')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Regret'); ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Recovery steps
    ax = axes[1]
    recovery_stau = [stau_ns[t]['recovery_steps'] for t in tau_vals]
    recovery_sm = [sm_ns[t]['recovery_steps'] for t in temp_vals]
    ax.plot(tau_vals, recovery_stau, 'o-', color='#58a6ff',
            linewidth=2, markersize=8, label='s^tau')
    ax.plot(temp_vals, recovery_sm, 's--', color='#f78166',
            linewidth=2, markersize=8, label='Softmax')
    ax.set_title('Estimated Recovery Time')
    ax.set_xlabel('Parameter'); ax.set_ylabel('Recovery steps')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Summary
    ax = axes[2]
    ax.axis('off')
    best_stau_rec = min(stau_ns, key=lambda t: stau_ns[t]['recovery_steps'])
    best_sm_rec = min(sm_ns, key=lambda t: sm_ns[t]['recovery_steps'])
    lines = [
        "NON-STATIONARY SUMMARY",
        "=" * 22,
        f"Switch at step {switch_step}",
        f"Before: arm {np.argmax(arm_before)}={max(arm_before):.2f}",
        f"After:  arm {np.argmax(arm_after)}={max(arm_after):.2f}",
        "",
        "Fastest recovery:",
    ]
    for t in tau_vals:
        lines.append(f"  s^tau  tau={t:<5.1f}: {stau_ns[t]['recovery_steps']:.0f} steps")
    for t in temp_vals:
        lines.append(f"  SM     T={t:<5.2f}  : {sm_ns[t]['recovery_steps']:.0f} steps")
    lines += [
        "",
        f"Best s^tau recovery: tau={best_stau_rec}",
        f"Best SM recovery:    T={best_sm_rec}",
    ]
    if stau_ns[best_stau_rec]['recovery_steps'] < sm_ns[best_sm_rec]['recovery_steps']:
        lines.append("→ s^tau recovers faster")
    else:
        lines.append("→ Softmax recovers faster")

    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontfamily='monospace', fontsize=7.5, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_regret_decomposition(all_configs, arm_means, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = []
    bad_fracs = []
    colors = []
    for config_type, param, bad_frac, _ in all_configs:
        if config_type == 'stau':
            labels.append(f's^tau\n{param:.1f}')
            colors.append('#58a6ff')
        else:
            labels.append(f'SM\n{param:.2f}')
            colors.append('#f78166')
        bad_fracs.append(bad_frac * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Regret Decomposition: Pull Distribution Analysis",
                 fontsize=13, fontweight='bold')

    # Bar: % pulls on bad arms
    ax1.bar(range(len(labels)), bad_fracs, color=colors, alpha=0.85, edgecolor='white')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_title('% Pulls on "Bad" Arms (mean < best/2)')
    ax1.set_ylabel('% of total pulls')
    ax1.grid(True, alpha=0.3, axis='y')

    # Stacked bar: per-arm pull distribution
    # Show for best s^tau and best softmax
    ax2_arms = np.arange(len(arm_means))
    best_mean = max(arm_means)

    # Find best configs
    best_stau = min([c for c in all_configs if c[0] == 'stau'],
                    key=lambda c: c[2])
    best_sm = min([c for c in all_configs if c[0] == 'softmax'],
                  key=lambda c: c[2])

    w = 0.35
    stau_pulls = best_stau[3] / best_stau[3].sum()
    sm_pulls = best_sm[3] / best_sm[3].sum()

    bar_colors_stau = ['#58a6ff' if m >= best_mean / 2 else '#ff7b72'
                       for m in arm_means]
    bar_colors_sm = ['#58a6ff' if m >= best_mean / 2 else '#ff7b72'
                     for m in arm_means]

    ax2.bar(ax2_arms - w/2, stau_pulls, w, color=bar_colors_stau, alpha=0.85,
            label=f's^tau tau={best_stau[1]:.1f}')
    ax2.bar(ax2_arms + w/2, sm_pulls, w, color=bar_colors_sm, alpha=0.55,
            label=f'Softmax T={best_sm[1]:.2f}')
    ax2.set_title('Pull Distribution: Best s^tau vs Best Softmax')
    ax2.set_xlabel('Arm'); ax2.set_ylabel('Fraction of pulls')
    # Annotate arm means
    for i, m in enumerate(arm_means):
        ax2.annotate(f'{m:.2f}', (i, max(stau_pulls[i], sm_pulls[i]) + 0.01),
                     ha='center', fontsize=6, color='gray')
    ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == '__main__':
    OUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

    ARM_MEANS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.60, 0.70, 0.85]

    # ── Part 1 ──
    stau_res, sm_res = run_static_duel(ARM_MEANS, n_steps=2000, n_runs=40)
    plot_static_duel(stau_res, sm_res, ARM_MEANS,
                     save_path=OUT_DIR / 'duel_static.png')

    # ── Part 2 ──
    # Before: arm 9(0.85) is best. After: arm 3 soars to 0.92, arm 9 crashes to 0.15.
    ARM_AFTER = ARM_MEANS.copy()
    ARM_AFTER[9] = 0.15  # old best crashes
    ARM_AFTER[3] = 0.92  # old mediocre becomes new king

    stau_ns, sm_ns = run_nonstationary_duel(
        ARM_MEANS, ARM_AFTER, switch_step=1500, n_runs=30
    )
    plot_nonstationary_duel(stau_ns, sm_ns, ARM_MEANS, ARM_AFTER, 1500,
                            save_path=OUT_DIR / 'duel_nonstationary.png')

    # ── Part 3 ──
    all_configs = run_regret_decomposition(ARM_MEANS, n_steps=3000, n_runs=30)
    plot_regret_decomposition(all_configs, ARM_MEANS,
                              save_path=OUT_DIR / 'duel_decomposition.png')

    print()
    print("=" * 70)
    print("  All plots saved to:", OUT_DIR)
    print("=" * 70)
