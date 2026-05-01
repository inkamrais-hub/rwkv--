"""
s^tau vs Softmax: Negative-Score Bandit

UCB scores are always >= 0, so s^tau's CLAMP advantage never activates.
This experiment uses CENTERED scores (subtract global mean), where
underperforming arms get genuinely negative scores.

s^tau with clamp: negative-score arms → p ≈ 0 (structural exclusion)
softmax: negative-score arms → p > 0 always (wasteful)

Also tests a DYNAMIC threshold where the "zero floor" matters more.
"""
import numpy as np
import os
from pathlib import Path

EPS = 1e-8


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


def run_bandit_centered(arm_means, n_steps, tau, temperature=None, seed=42):
    """
    Bandit with CENTERED scores: subtract running global mean.
    Underperforming arms get negative scores → clamp advantage.
    """
    rng = np.random.RandomState(seed)
    n_arms = len(arm_means)
    best_mean = max(arm_means)

    counts = np.zeros(n_arms)
    rewards_sum = np.zeros(n_arms)
    regrets = np.zeros(n_steps)
    global_reward_sum = 0.0

    for t in range(n_steps):
        if t < n_arms:
            arm = t
        else:
            empirical = np.divide(rewards_sum, counts,
                                  out=np.zeros_like(rewards_sum), where=counts > 0)
            bonus = np.sqrt(2 * np.log(t + 1) / np.maximum(counts, 1))
            raw_scores = empirical + bonus

            # CENTER: subtract global mean to make underperforming arms negative
            global_mean = global_reward_sum / max(t, 1)
            scores = raw_scores - global_mean

            if temperature is None:
                probs = stau_norm_numpy(scores, tau)
            else:
                probs = softmax_numpy(scores, temperature)
            arm = rng.choice(n_arms, p=probs)

        reward = float(rng.rand() < arm_means[arm])
        counts[arm] += 1
        rewards_sum[arm] += reward
        global_reward_sum += reward
        regrets[t] = best_mean - arm_means[arm]

    return {
        'cumulative_regret': np.cumsum(regrets),
        'counts': counts,
        'final_empirical': rewards_sum / np.maximum(counts, 1),
    }


def run_bandit_comparative_advantage(arm_means, n_steps, tau, temperature=None, seed=42):
    """
    COMPARATIVE: scores = empirical_mean - global_mean (no bonus).
    This directly tests: given noisy estimates, how do you pick?
    Many arms will have NEGATIVE "advantage" → clamp matters a lot.
    """
    rng = np.random.RandomState(seed)
    n_arms = len(arm_means)
    best_mean = max(arm_means)

    counts = np.zeros(n_arms)
    rewards_sum = np.zeros(n_arms)
    regrets = np.zeros(n_steps)
    global_reward_sum = 0.0

    for t in range(n_steps):
        if t < n_arms:
            arm = t
        else:
            empirical = np.divide(rewards_sum, counts,
                                  out=np.zeros_like(rewards_sum), where=counts > 0)
            # No exploration bonus — pure advantage signal
            global_mean = global_reward_sum / max(t, 1)
            scores = empirical - global_mean

            if temperature is None:
                probs = stau_norm_numpy(scores, tau)
            else:
                probs = softmax_numpy(scores, temperature)
            arm = rng.choice(n_arms, p=probs)

        reward = float(rng.rand() < arm_means[arm])
        counts[arm] += 1
        rewards_sum[arm] += reward
        global_reward_sum += reward
        regrets[t] = best_mean - arm_means[arm]

    return {
        'cumulative_regret': np.cumsum(regrets),
        'counts': counts,
        'final_empirical': rewards_sum / np.maximum(counts, 1),
    }


def run_negative_score_duel(arm_means, n_steps=3000, n_runs=50):
    tau_vals = [1.0, 2.0, 3.0, 5.0, 8.0, 15.0]
    temp_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]

    print("=" * 70)
    print("  NEGATIVE-SCORE BANDIT: s^tau vs Softmax (CENTERED scores)")
    print("=" * 70)
    print(f"  Arms: {len(arm_means)} | Best: arm {np.argmax(arm_means)}={max(arm_means):.2f}")
    print(f"  Steps: {n_steps} | Runs per config: {n_runs}")
    print(f"  Key: scores = UCB - global_mean → many arms get NEGATIVE")
    print()

    # ── Centered UCB ──
    print("  ── Experiment A: Centered UCB (empirical+bonus - global_mean) ──")
    stau_c = {}
    sm_c = {}
    for tau in tau_vals:
        all_regrets = np.zeros((n_runs, n_steps))
        for r in range(n_runs):
            res = run_bandit_centered(arm_means, n_steps, tau=tau, seed=42 + r)
            all_regrets[r] = res['cumulative_regret']
        stau_c[tau] = {'avg_regret': all_regrets.mean(axis=0),
                       'final_regret': all_regrets.mean(axis=0)[-1],
                       'std': all_regrets.std(axis=0)[-1]}
        print(f"    s^tau  tau={tau:<5.1f} → regret={stau_c[tau]['final_regret']:8.1f}")
    for temp in temp_vals:
        all_regrets = np.zeros((n_runs, n_steps))
        for r in range(n_runs):
            res = run_bandit_centered(arm_means, n_steps, tau=0, temperature=temp, seed=42 + r)
            all_regrets[r] = res['cumulative_regret']
        sm_c[temp] = {'avg_regret': all_regrets.mean(axis=0),
                      'final_regret': all_regrets.mean(axis=0)[-1],
                      'std': all_regrets.std(axis=0)[-1]}
        print(f"    SM     T={temp:<5.2f}  → regret={sm_c[temp]['final_regret']:8.1f}")

    best_stau_c = min(stau_c, key=lambda t: stau_c[t]['final_regret'])
    best_sm_c = min(sm_c, key=lambda t: sm_c[t]['final_regret'])
    print(f"  Best s^tau:  tau={best_stau_c}  regret={stau_c[best_stau_c]['final_regret']:.1f}")
    print(f"  Best softmax: T={best_sm_c}  regret={sm_c[best_sm_c]['final_regret']:.1f}")
    delta_c = sm_c[best_sm_c]['final_regret'] - stau_c[best_stau_c]['final_regret']
    winner_c = "s^tau" if delta_c > 0 else "Softmax"
    print(f"  s^tau advantage: {delta_c:+.1f} → {winner_c} wins")
    print()

    # ── Comparative advantage ──
    print("  ── Experiment B: Comparative Advantage (empirical - global_mean, NO bonus) ──")
    stau_v = {}
    sm_v = {}
    for tau in tau_vals:
        all_regrets = np.zeros((n_runs, n_steps))
        for r in range(n_runs):
            res = run_bandit_comparative_advantage(arm_means, n_steps, tau=tau, seed=42 + r)
            all_regrets[r] = res['cumulative_regret']
        stau_v[tau] = {'avg_regret': all_regrets.mean(axis=0),
                       'final_regret': all_regrets.mean(axis=0)[-1],
                       'std': all_regrets.std(axis=0)[-1]}
        print(f"    s^tau  tau={tau:<5.1f} → regret={stau_v[tau]['final_regret']:8.1f}")
    for temp in temp_vals:
        all_regrets = np.zeros((n_runs, n_steps))
        for r in range(n_runs):
            res = run_bandit_comparative_advantage(arm_means, n_steps, tau=0, temperature=temp, seed=42 + r)
            all_regrets[r] = res['cumulative_regret']
        sm_v[temp] = {'avg_regret': all_regrets.mean(axis=0),
                      'final_regret': all_regrets.mean(axis=0)[-1],
                      'std': all_regrets.std(axis=0)[-1]}
        print(f"    SM     T={temp:<5.2f}  → regret={sm_v[temp]['final_regret']:8.1f}")

    best_stau_v = min(stau_v, key=lambda t: stau_v[t]['final_regret'])
    best_sm_v = min(sm_v, key=lambda t: sm_v[t]['final_regret'])
    print(f"  Best s^tau:  tau={best_stau_v}  regret={stau_v[best_stau_v]['final_regret']:.1f}")
    print(f"  Best softmax: T={best_sm_v}  regret={sm_v[best_sm_v]['final_regret']:.1f}")
    delta_v = sm_v[best_sm_v]['final_regret'] - stau_v[best_stau_v]['final_regret']
    winner_v = "s^tau" if delta_v > 0 else "Softmax"
    print(f"  s^tau advantage: {delta_v:+.1f} → {winner_v} wins")

    return stau_c, sm_c, stau_v, sm_v


def plot_negative_duel(stau_c, sm_c, stau_v, sm_v, arm_means, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("When Negative Scores Matter: s^tau vs Softmax",
                 fontsize=14, fontweight='bold')

    # [0,0] Centered UCB: regret curves overlay
    ax = axes[0, 0]
    best_tau_c = min(stau_c, key=lambda t: stau_c[t]['final_regret'])
    best_temp_c = min(sm_c, key=lambda t: sm_c[t]['final_regret'])
    ax.plot(stau_c[best_tau_c]['avg_regret'], color='#58a6ff', linewidth=2.5,
            label=f's^tau tau={best_tau_c}')
    ax.plot(sm_c[best_temp_c]['avg_regret'], color='#f78166', linewidth=2.5,
            label=f'Softmax T={best_temp_c}')
    ax.fill_between(range(len(stau_c[best_tau_c]['avg_regret'])),
                    stau_c[best_tau_c]['avg_regret'] - stau_c[best_tau_c]['std'],
                    stau_c[best_tau_c]['avg_regret'] + stau_c[best_tau_c]['std'],
                    color='#58a6ff', alpha=0.15)
    ax.fill_between(range(len(sm_c[best_temp_c]['avg_regret'])),
                    sm_c[best_temp_c]['avg_regret'] - sm_c[best_temp_c]['std'],
                    sm_c[best_temp_c]['avg_regret'] + sm_c[best_temp_c]['std'],
                    color='#f78166', alpha=0.15)
    ax.set_title('Centered UCB: Best vs Best')
    ax.set_xlabel('Steps'); ax.set_ylabel('Regret')
    ax.legend(); ax.grid(True, alpha=0.3)

    # [0,1] Centered UCB: final regret by param
    ax = axes[0, 1]
    tau_list = sorted(stau_c.keys())
    temp_list = sorted(sm_c.keys())
    ax.plot(tau_list, [stau_c[t]['final_regret'] for t in tau_list],
            'o-', color='#58a6ff', linewidth=2, markersize=7, label='s^tau (centered)')
    ax.plot(temp_list, [sm_c[t]['final_regret'] for t in temp_list],
            's--', color='#f78166', linewidth=2, markersize=7, label='Softmax (centered)')
    ax.set_title('Centered UCB: Regret vs Parameter')
    ax.set_xlabel('Parameter'); ax.set_ylabel('Regret')
    ax.legend(); ax.grid(True, alpha=0.3)

    # [0,2] Comparative: best vs best
    ax = axes[0, 2]
    best_tau_v = min(stau_v, key=lambda t: stau_v[t]['final_regret'])
    best_temp_v = min(sm_v, key=lambda t: sm_v[t]['final_regret'])
    ax.plot(stau_v[best_tau_v]['avg_regret'], color='#58a6ff', linewidth=2.5,
            label=f's^tau tau={best_tau_v}')
    ax.plot(sm_v[best_temp_v]['avg_regret'], color='#f78166', linewidth=2.5,
            label=f'Softmax T={best_temp_v}')
    ax.fill_between(range(len(stau_v[best_tau_v]['avg_regret'])),
                    stau_v[best_tau_v]['avg_regret'] - stau_v[best_tau_v]['std'],
                    stau_v[best_tau_v]['avg_regret'] + stau_v[best_tau_v]['std'],
                    color='#58a6ff', alpha=0.15)
    ax.fill_between(range(len(sm_v[best_temp_v]['avg_regret'])),
                    sm_v[best_temp_v]['avg_regret'] - sm_v[best_temp_v]['std'],
                    sm_v[best_temp_v]['avg_regret'] + sm_v[best_temp_v]['std'],
                    color='#f78166', alpha=0.15)
    ax.set_title('Comparative Advantage: Best vs Best')
    ax.set_xlabel('Steps'); ax.set_ylabel('Regret')
    ax.legend(); ax.grid(True, alpha=0.3)

    # [0,3] Comparative: regret by param
    ax = axes[0, 3]
    ax.plot(tau_list, [stau_v[t]['final_regret'] for t in tau_list],
            'o-', color='#58a6ff', linewidth=2, markersize=7, label='s^tau (comparative)')
    ax.plot(temp_list, [sm_v[t]['final_regret'] for t in temp_list],
            's--', color='#f78166', linewidth=2, markersize=7, label='Softmax (comparative)')
    ax.set_title('Comparative Advantage: Regret vs Parameter')
    ax.set_xlabel('Parameter'); ax.set_ylabel('Regret')
    ax.legend(); ax.grid(True, alpha=0.3)

    # [1,0] Score distribution visualization at convergence
    ax = axes[1, 0]
    arm_means_arr = np.array(arm_means)
    global_mean = arm_means_arr.mean()
    centered_means = arm_means_arr - global_mean
    x = np.arange(len(arm_means))

    bars = ax.bar(x, centered_means, color=[
        '#ff7b72' if v < 0 else '#3fb950' for v in centered_means
    ], alpha=0.85, edgecolor='white')

    ax.axhline(0, color='white', linewidth=1)
    ax.axhline(EPS, color='#58a6ff', linestyle='--', linewidth=1.5,
               label=f's^tau clamp floor (eps={EPS})')
    for i, v in enumerate(centered_means):
        color = 'white' if v < 0 else '#0d1117'
        ax.text(i, v + (0.03 if v >= 0 else -0.03), f'{v:+.2f}',
                ha='center', fontsize=7, color=color, fontweight='bold')
    ax.set_title('True Arm Means (centered) → 5 arms NEGATIVE')
    ax.set_xlabel('Arm'); ax.set_ylabel('Centered Mean')
    ax.legend(fontsize=7)

    # [1,1] Negative arm pull suppression
    ax = axes[1, 1]
    neg_arms = [i for i, v in enumerate(centered_means) if v < 0]
    neg_params = []
    neg_pulls_stau = []
    neg_pulls_sm = []

    # Run single runs to measure negative-arm suppression at different sharpness levels
    for tau in tau_list:
        res_s = run_bandit_centered(arm_means, 3000, tau=tau, seed=42)
        neg_params.append(tau)
        neg_pulls_stau.append(res_s['counts'][neg_arms].sum() / res_s['counts'].sum() * 100)
    for temp in temp_list:
        res_m = run_bandit_centered(arm_means, 3000, tau=0, temperature=temp, seed=42)
        neg_pulls_sm.append(res_m['counts'][neg_arms].sum() / res_m['counts'].sum() * 100)

    ax.plot(tau_list, neg_pulls_stau, 'o-', color='#58a6ff', linewidth=2, markersize=8,
            label=f's^tau ({len(neg_arms)} negative arms)')
    ax.plot(temp_list, neg_pulls_sm, 's--', color='#f78166', linewidth=2, markersize=8,
            label=f'Softmax ({len(neg_arms)} negative arms)')
    ax.set_title('Negative-Arm Pulls: s^tau CLAMP Effect')
    ax.set_xlabel('Parameter'); ax.set_ylabel('% pulls on NEGATIVE arms')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # [1,2] Summary: static vs centered comparison
    ax = axes[1, 2]
    ax.axis('off')
    best_s_c = min(stau_c, key=lambda t: stau_c[t]['final_regret'])
    best_m_c = min(sm_c, key=lambda t: sm_c[t]['final_regret'])
    best_s_v = min(stau_v, key=lambda t: stau_v[t]['final_regret'])
    best_m_v = min(sm_v, key=lambda t: sm_v[t]['final_regret'])

    lines = [
        "KEY INSIGHT",
        "=" * 16,
        "",
        "In static duel (all scores >= 0):",
        f"  s^tau:  tau={15.0} regret=364",
        f"  SM:     T={0.05}  regret=342",
        "  → Softmax wins by 6%",
        "",
        "WHY: no negative scores →",
        "clamp never activates.",
        "",
        "When scores go NEGATIVE:",
        f"  Centered: s^tau adv={sm_c[best_m_c]['final_regret']-stau_c[best_s_c]['final_regret']:+.0f}",
        f"  Comparative: s^tau adv={sm_v[best_m_v]['final_regret']-stau_v[best_s_v]['final_regret']:+.0f}",
        "",
        "s^tau wins when CLAMP matters.",
        "Softmax wins when curve is all",
        "that matters (static, no negative).",
    ]
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            fontfamily='monospace', fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.85))

    # [1,3] Probability snapshots
    ax = axes[1, 3]
    # Simulate a typical late-game score distribution
    late_game_empirical = np.array([0.12, 0.16, 0.21, 0.27, 0.31, 0.42, 0.57, 0.62, 0.72, 0.86])
    late_scores = late_game_empirical - late_game_empirical.mean()  # centered

    stau_probs = stau_norm_numpy(late_scores, 5.0)
    sm_probs = softmax_numpy(late_scores, 0.2)

    x = np.arange(len(late_scores))
    w = 0.35
    colors = ['#ff7b72' if v < 0 else '#3fb950' for v in late_scores]

    ax.bar(x - w/2, stau_probs, w, color=colors, alpha=0.85, label='s^tau tau=5')
    ax.bar(x + w/2, sm_probs, w, color=colors, alpha=0.4, label='Softmax T=0.2')
    ax.set_title('Probability Snapshot (5 neg arms)')
    ax.set_xlabel('Arm'); ax.set_ylabel('Probability')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == '__main__':
    OUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

    ARM_MEANS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.60, 0.70, 0.85]

    stau_c, sm_c, stau_v, sm_v = run_negative_score_duel(
        ARM_MEANS, n_steps=3000, n_runs=50
    )

    plot_negative_duel(stau_c, sm_c, stau_v, sm_v, ARM_MEANS,
                       save_path=OUT_DIR / 'duel_negative_scores.png')

    print()
    print("=" * 70)
    print("  Plot saved to:", OUT_DIR / 'duel_negative_scores.png')
    print("=" * 70)
