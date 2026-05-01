import numpy as np
import os, sys

# Allow importing s_tau from deploy_pkg when run from experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deploy_pkg'))
_s_tau_available = True
try:
    import torch
    from attention_mechanisms.s_tau_fused import s_tau_norm
except Exception:
    _s_tau_available = False

EPS = 1e-8

def stau_norm_numpy(scores, tau):
    clamped = np.maximum(scores, EPS)
    powered = np.power(clamped, tau)
    s = powered.sum()
    return powered / s if s > 0 else np.ones_like(powered) / len(powered)


def run_bandit_experiment(
    arm_means,
    n_steps=1000,
    tau=2.0,
    eps=1e-8,
    seed=42,
):
    """
    Multi-armed bandit with s^tau attention as decision mechanism.

    At each step:
    1. Compute "attention score" for each arm = empirical_mean + uncertainty_bonus
    2. Apply s^tau normalization to get a probability distribution over arms
    3. Sample an arm from this distribution (soft action selection)
    4. Pull arm, observe reward, update running statistics

    tau controls explore/exploit:
    - tau << 1: flat distribution -> heavy exploration
    - tau = 1: L1 normalization -> moderate selection
    - tau >> 1: sharp distribution -> aggressive exploitation
    """
    rng = np.random.RandomState(seed)
    n_arms = len(arm_means)
    best_mean = max(arm_means)

    counts = np.zeros(n_arms)
    rewards_sum = np.zeros(n_arms)
    regrets = np.zeros(n_steps)
    chosen = np.zeros(n_steps, dtype=int)
    reward_history = np.zeros(n_steps)

    for t in range(n_steps):
        # UCB-style score: empirical mean + exploration bonus
        if t < n_arms:
            # Initial round-robin: pull each arm once
            arm = t
        else:
            empirical = np.divide(rewards_sum, counts,
                                  out=np.zeros_like(rewards_sum), where=counts > 0)
            # Exploration bonus decays with 1/sqrt(count)
            bonus = np.sqrt(2 * np.log(t + 1) / np.maximum(counts, 1))
            scores = empirical + bonus

            probs = stau_norm_numpy(scores, tau)
            arm = rng.choice(n_arms, p=probs)

        reward = float(rng.rand() < arm_means[arm])
        counts[arm] += 1
        rewards_sum[arm] += reward
        reward_history[t] = reward
        chosen[t] = arm

        # Instantaneous regret: difference between best arm and chosen arm
        regrets[t] = best_mean - arm_means[arm]

    cumulative_regret = np.cumsum(regrets)
    return {
        'cumulative_regret': cumulative_regret,
        'chosen': chosen,
        'rewards': reward_history,
        'counts': counts,
        'final_empirical': rewards_sum / np.maximum(counts, 1),
    }


def run_comparison(
    arm_means=None,
    n_steps=1000,
    tau_values=None,
    n_runs=20,
    seed=42,
):
    if arm_means is None:
        arm_means = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.55, 0.6, 0.7, 0.85]

    if tau_values is None:
        tau_values = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 20.0]

    print("=" * 70)
    print("  s^tau Multi-Armed Bandit: Explore/Exploit Trade-off")
    print("=" * 70)
    print(f"  Arms: {len(arm_means)} | Best mean: {max(arm_means):.2f}")
    print(f"  Steps per run: {n_steps} | Runs per tau: {n_runs}")
    print(f"  Tau values: {tau_values}")
    print()

    results = {}
    for tau in tau_values:
        all_regrets = np.zeros((n_runs, n_steps))
        all_final_empirical = np.zeros((n_runs, len(arm_means)))
        all_accuracy = np.zeros(n_runs)

        for run in range(n_runs):
            res = run_bandit_experiment(
                arm_means, n_steps=n_steps, tau=tau, seed=seed + run
            )
            all_regrets[run] = res['cumulative_regret']
            all_final_empirical[run] = res['final_empirical']

            # Accuracy: how often the most-pulled arm is the true best
            best_arm = np.argmax(arm_means)
            most_pulled = np.argmax(res['counts'])
            all_accuracy[run] = float(most_pulled == best_arm)

        avg_regret = all_regrets.mean(axis=0)
        std_regret = all_regrets.std(axis=0)
        final_regret = avg_regret[-1]
        accuracy = all_accuracy.mean()

        results[tau] = {
            'avg_regret': avg_regret,
            'std_regret': std_regret,
            'final_regret': final_regret,
            'accuracy': accuracy,
            'final_empirical': all_final_empirical.mean(axis=0),
        }

        print(f"  tau={tau:5.1f} | Final Regret: {final_regret:8.2f} "
              f"±{std_regret[-1]:.2f} | Best-arm Accuracy: {accuracy:.1%}")

    print()
    print("  ──────────────────────────────────────────────────────")
    best_tau = min(results, key=lambda t: results[t]['final_regret'])
    best_acc_tau = max(results, key=lambda t: results[t]['accuracy'])
    print(f"  Lowest regret: tau={best_tau} ({results[best_tau]['final_regret']:.1f})")
    print(f"  Highest accuracy: tau={best_acc_tau} ({results[best_acc_tau]['accuracy']:.1%})")
    print()

    return results, arm_means


def plot_results(results, arm_means, save_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tau_list = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(tau_list)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("s^τ Multi-Armed Bandit: Explore/Exploit Trade-off",
                 fontsize=14, fontweight='bold', y=0.98)

    # [0,0] Cumulative regret over time
    ax = axes[0, 0]
    for tau, c in zip(tau_list, colors):
        r = results[tau]['avg_regret']
        ax.plot(r, color=c, alpha=0.9, linewidth=1.5,
                label=f'τ={tau:.1f}')
    ax.set_title('Average Cumulative Regret')
    ax.set_xlabel('Steps'); ax.set_ylabel('Cumulative Regret')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)

    # [0,1] Final regret vs tau
    ax = axes[0, 1]
    final_regrets = [results[t]['final_regret'] for t in tau_list]
    ax.plot(tau_list, final_regrets, 'o-', color='#58a6ff',
            markersize=8, linewidth=2)
    best_tau = tau_list[np.argmin(final_regrets)]
    ax.axvline(best_tau, color='#f78166', linestyle='--', alpha=0.7,
               label=f'Best τ={best_tau}')
    ax.set_title('Final Regret vs τ')
    ax.set_xlabel('τ'); ax.set_ylabel('Cumulative Regret')
    ax.legend(); ax.grid(True, alpha=0.3)

    # [0,2] Best-arm accuracy vs tau
    ax = axes[0, 2]
    accuracies = [results[t]['accuracy'] * 100 for t in tau_list]
    ax.plot(tau_list, accuracies, 's-', color='#3fb950',
            markersize=8, linewidth=2)
    ax.set_title('Best-Arm Identification Accuracy')
    ax.set_xlabel('τ'); ax.set_ylabel('Accuracy (%)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # [1,0] Regret heatmap for a single run
    ax = axes[1, 0]
    regret_matrix = np.array([results[t]['avg_regret'] for t in tau_list])
    im = ax.imshow(regret_matrix, aspect='auto', cmap='YlOrRd',
                   extent=[0, regret_matrix.shape[1],
                           tau_list[0], tau_list[-1]],
                   origin='lower')
    ax.set_title('Regret over Steps (τ vs Step)')
    ax.set_xlabel('Step'); ax.set_ylabel('τ')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Cumulative Regret')

    # [1,1] Arm selection distribution
    ax = axes[1, 1]
    x = np.arange(len(arm_means))
    width = 0.8 / len(arm_means)
    best_arm = np.argmax(arm_means)
    for i, tau in enumerate([tau_list[0], tau_list[len(tau_list)//2], tau_list[-1]]):
        res = results[tau]
        emp = res['final_empirical']
        if isinstance(emp, np.ndarray):
            ax.bar(x + i * width - 0.4 + width/2, emp, width,
                   color=colors[[tau_list.index(t) for t in tau_list].index(i)],
                   alpha=0.85, label=f'τ={tau:.1f} (learned)')
    ax.axvline(best_arm, color='#ff7b72', linestyle='--', linewidth=2,
               label=f'True Best (arm {best_arm})')
    ax.set_title('Learned Arm Values at Final Step')
    ax.set_xlabel('Arm'); ax.set_ylabel('Empirical Mean')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # [1,2] Summary text
    ax = axes[1, 2]
    ax.axis('off')
    lines = [
        "Summary",
        "=" * 20,
        f"Arms: {len(arm_means)}",
        f"Best arm mean: {max(arm_means):.2f}",
        f"Best arm idx: {best_arm}",
        "",
        "τ Effect:",
        "  Low τ → flat → explore",
        "  High τ → sharp → exploit",
        "  Optimal τ balances both",
        "",
    ]
    for t in tau_list:
        lines.append(f"  τ={t:5.1f}: regret={results[t]['final_regret']:7.1f} "
                     f"acc={results[t]['accuracy']:.1%}")

    text = '\n'.join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"\n  Figure saved to: {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    ARM_MEANS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.60, 0.70, 0.85]
    TAU_VALUES = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 20.0]
    N_STEPS = 2000
    N_RUNS = 30

    results, arm_means = run_comparison(
        ARM_MEANS, n_steps=N_STEPS, tau_values=TAU_VALUES, n_runs=N_RUNS
    )

    try:
        save_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(save_dir, 'tau_bandit_results.png')
        plot_results(results, ARM_MEANS, save_path=save_path)
    except ImportError:
        print("\n  [info] matplotlib not available; skipping plot.")
        print("  [info] Install with: pip install matplotlib")
