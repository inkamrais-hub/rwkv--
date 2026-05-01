"""
s_tau_lab — shared building blocks for τ experiments

Pure numpy, zero model dependency. Importable from any experiment script.

Core:
  stau_norm(scores, tau)  — s^τ normalization (clamp + pow + normalize)
  softmax_norm(scores, T) — softmax with temperature
  entropy(probs)          — attention entropy
  effective_n(probs)      — effective number (1/sum(p^2))

Bandit:
  BanditEnv               — N-armed bandit with configurable means
  run_bandit              — single bandit run, policy-agnostic
  bandit_sweep            — parameter sweep across tau/temperature

Plotting (matplotlib):
  plot_regret_curves      — cumulative regret overlay
  plot_final_regret       — final regret vs parameter
  plot_prob_snapshot      — bar chart of probability distribution
  save_or_show            — save to path or plt.show()
"""
import numpy as np

EPS = 1e-8

# ═══════════════════════════════════════
#  Normalization
# ═══════════════════════════════════════

def stau_norm(scores, tau):
    """s^τ: a_i = clamp(s_i, ε)^τ / Σ clamp(s_j, ε)^τ"""
    scores = np.asarray(scores, dtype=np.float64)
    c = np.maximum(scores, EPS)
    p = np.power(c, tau)
    s = p.sum(axis=-1, keepdims=True)
    return np.divide(p, s, where=s > 0,
                     out=np.ones_like(p) / p.shape[-1])


def softmax_norm(scores, temperature=1.0):
    """Softmax with temperature (numerically stable)"""
    scores = np.asarray(scores, dtype=np.float64)
    s = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(s / max(temperature, 0.01))
    return e / e.sum(axis=-1, keepdims=True)


def decide(scores, tau=None, temperature=None, rng=None):
    """Sample from s^tau or softmax distribution. Returns (index, probs)."""
    if rng is None:
        rng = np.random
    if temperature is not None:
        probs = softmax_norm(scores, temperature)
    else:
        probs = stau_norm(scores, tau)
    flat = probs.flatten()
    return rng.choice(len(flat), p=flat), flat

# ═══════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════

def entropy(probs):
    """Entropy of probability distribution (bits)"""
    probs = np.asarray(probs)
    return -np.sum(probs * np.log2(np.maximum(probs, 1e-12)), axis=-1)


def effective_n(probs):
    """Effective number: 1 / sum(p^2). Higher = more uniform."""
    probs = np.asarray(probs)
    sq = (probs * probs).sum(axis=-1)
    return np.divide(1.0, sq, where=sq > 0, out=np.zeros_like(sq))


def max_weight(probs):
    """Maximum probability value"""
    return probs.max(axis=-1)


# ═══════════════════════════════════════
#  Bandit Engine
# ═══════════════════════════════════════

class BanditEnv:
    """N-armed bandit with Bernoulli rewards."""
    def __init__(self, arm_means, seed=42):
        self.arm_means = np.asarray(arm_means)
        self.n_arms = len(arm_means)
        self.best_mean = self.arm_means.max()
        self.rng = np.random.RandomState(seed)

    def best_arm(self):
        return int(np.argmax(self.arm_means))

    def pull(self, arm):
        return float(self.rng.rand() < self.arm_means[arm])


def run_bandit(env, n_steps, tau=None, temperature=None, seed=42,
               score_fn=None):
    """
    Generic bandit run with UCB-style scoring.

    score_fn: (empirical, counts, t) -> scores array. Default: UCB1.
    Returns dict with cumulative_regret, counts, chosen, rewards.
    """
    rng = np.random.RandomState(seed)
    n_arms = env.n_arms
    best_mean = env.best_mean

    if score_fn is None:
        def score_fn(emp, cts, t):
            bonus = np.sqrt(2 * np.log(t + 1) / np.maximum(cts, 1))
            return emp + bonus

    counts = np.zeros(n_arms)
    rewards_sum = np.zeros(n_arms)
    regrets = np.zeros(n_steps)
    chosen = np.zeros(n_steps, dtype=int)
    rewards = np.zeros(n_steps)

    for t in range(n_steps):
        if t < n_arms:
            arm = t
        else:
            empirical = np.divide(rewards_sum, counts,
                                  out=np.zeros_like(rewards_sum), where=counts > 0)
            scores = score_fn(empirical, counts, t)
            arm, _ = decide(scores.reshape(1, -1), tau=tau, temperature=temperature, rng=rng)
            arm = int(arm)

        reward = env.pull(arm)
        counts[arm] += 1
        rewards_sum[arm] += reward
        rewards[t] = reward
        chosen[t] = arm
        regrets[t] = best_mean - env.arm_means[arm]

    return {
        'cumulative_regret': np.cumsum(regrets),
        'counts': counts,
        'chosen': chosen,
        'rewards': rewards,
        'final_empirical': rewards_sum / np.maximum(counts, 1),
    }


def bandit_sweep(env, n_steps, n_runs, tau_vals=None, temp_vals=None, seed=42):
    """Sweep tau and temperature values. Returns dict of aggregated results."""
    results = {}
    if tau_vals:
        for tau in tau_vals:
            all_regrets = np.zeros((n_runs, n_steps))
            for r in range(n_runs):
                res = run_bandit(env, n_steps, tau=tau, seed=seed + r)
                all_regrets[r] = res['cumulative_regret']
            avg = all_regrets.mean(axis=0)
            results[f'tau={tau}'] = {
                'avg_regret': avg,
                'final_regret': avg[-1],
                'std': all_regrets.std(axis=0)[-1],
            }
    if temp_vals:
        for temp in temp_vals:
            all_regrets = np.zeros((n_runs, n_steps))
            for r in range(n_runs):
                res = run_bandit(env, n_steps, temperature=temp, seed=seed + r)
                all_regrets[r] = res['cumulative_regret']
            avg = all_regrets.mean(axis=0)
            results[f'T={temp}'] = {
                'avg_regret': avg,
                'final_regret': avg[-1],
                'std': all_regrets.std(axis=0)[-1],
            }
    return results


# ═══════════════════════════════════════
#  Plotting helpers (matplotlib)
# ═══════════════════════════════════════

def _ensure_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_regret_curves(results_dict, title='Regret', save_path=None):
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, data in results_dict.items():
        ax.plot(data['avg_regret'], linewidth=2, label=label)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Cumulative Regret')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    _finish_plot(fig, save_path)


def plot_final_regret(param_vals, regrets_stau, regrets_sm,
                      title='Final Regret', save_path=None):
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(param_vals, regrets_stau, 'o-', color='#58a6ff',
            linewidth=2, markersize=8, label='s^tau')
    ax.plot(param_vals, regrets_sm, 's--', color='#f78166',
            linewidth=2, markersize=8, label='Softmax')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Parameter'); ax.set_ylabel('Regret')
    ax.legend(); ax.grid(True, alpha=0.3)
    _finish_plot(fig, save_path)


def plot_prob_snapshot(scores, tau, temperature=None, title=None, save_path=None):
    """Side-by-side: s^tau vs softmax probability from same scores."""
    plt = _ensure_matplotlib()
    stau_probs = stau_norm(scores, tau).flatten()
    sm_probs = softmax_norm(scores, temperature or 1.0).flatten()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(scores))

    neg_mask = scores < 0
    colors = ['#ff7b72' if n else '#3fb950' for n in neg_mask]

    ax1.bar(x, stau_probs, color=colors, alpha=0.85, edgecolor='white')
    for i, (p, s) in enumerate(zip(stau_probs, scores)):
        if p > 0.02:
            ax1.text(i, p + 0.01, f'{p:.1%}', ha='center', fontsize=7)
    ax1.set_title(f's^tau (tau={tau})', fontsize=12)
    ax1.set_xticks(x); ax1.set_xticklabels([f'{s:+.1f}' for s in scores], fontsize=7)
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x, sm_probs, color=colors, alpha=0.85, edgecolor='white')
    for i, (p, s) in enumerate(zip(sm_probs, scores)):
        if p > 0.02:
            ax2.text(i, p + 0.01, f'{p:.1%}', ha='center', fontsize=7)
    ax2.set_title(f'Softmax (T={temperature or 1.0})', fontsize=12)
    ax2.set_xticks(x); ax2.set_xticklabels([f'{s:+.1f}' for s in scores], fontsize=7)
    ax2.grid(True, alpha=0.3, axis='y')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    _finish_plot(fig, save_path)


def _finish_plot(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches='tight')
        print(f"  => {save_path}")
    else:
        import matplotlib.pyplot as plt
        plt.show()
    plt = _ensure_matplotlib()
    plt.close(fig)


# ═══════════════════════════════════════
#  Quick demo
# ═══════════════════════════════════════

if __name__ == '__main__':
    # Verify basics
    s = np.array([-2.0, -0.5, 0.0, 1.0, 3.0, 5.0])
    print("scores:", s)
    print("s^tau tau=1:", stau_norm(s, 1.0))
    print("s^tau tau=5:", stau_norm(s, 5.0))
    print("softmax T=1:", softmax_norm(s, 1.0))
    print("entropy(stau τ=5):", entropy(stau_norm(s, 5.0)))
    print("effective_n(stau τ=5):", effective_n(stau_norm(s, 5.0)))

    # Quick bandit
    env = BanditEnv([0.1, 0.3, 0.5, 0.7])
    res = run_bandit(env, 200, tau=3.0)
    print(f"Bandit regret: {res['cumulative_regret'][-1]:.1f}")
    print("All good.")
