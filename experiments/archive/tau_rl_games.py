"""
s^tau vs Softmax: Three RL Games (FIXED)

Each game now has REAL negative-score options that CLAMP should kill.

E1 — Grid: agent chooses 4 directions. Walls → -10, goal → +10, coin → +5.
     Wall directions = negative score → clamp should suppress.

E2 — Contextual: 10 signal features (mean +0.3) vs 10 noise features (mean -0.3).
     Noise features consistently negative → clamp should filter them out.

E3 — Poker: high noise (0.3) means scores jump around. -EV actions are risky.
     Clamp should resist temptation when EV clearly negative.
"""
import numpy as np
import os
from pathlib import Path

EPS = 1e-8
rng_state = np.random.RandomState(42)

def stau_norm(scores, tau):
    scores = np.asarray(scores, dtype=np.float64)
    clamped = np.maximum(scores, EPS)
    powered = np.power(clamped, tau)
    s = powered.sum(axis=-1, keepdims=True)
    return np.divide(powered, s, where=s > 0,
                     out=np.ones_like(powered) / powered.shape[-1])

def softmax_norm(scores, temperature=1.0):
    scores = np.asarray(scores, dtype=np.float64)
    s = scores - scores.max(axis=-1, keepdims=True)
    exps = np.exp(s / max(temperature, 0.01))
    return exps / exps.sum(axis=-1, keepdims=True)

def decide(scores, tau=None, temperature=None, rng=None):
    if rng is None:
        rng = np.random
    if temperature is not None:
        probs = softmax_norm(scores, temperature)
    else:
        probs = stau_norm(scores, tau)
    flat = probs.flatten()
    return rng.choice(len(flat), p=flat), flat


# ═══════════════════════════════════════════════════
#  E1: GRID NAVIGATION (fixed: 4 directions)
# ═══════════════════════════════════════════════════

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def build_grid_fixed(size=10, n_walls=15, n_coins=6, seed=42):
    rng = np.random.RandomState(seed)
    grid = np.zeros((size, size))
    grid[-1, -1] = 10.0  # goal

    walls = set()
    while len(walls) < n_walls:
        r, c = rng.randint(0, size), rng.randint(0, size)
        if (r, c) not in [(0, 0), (size-1, size-1)]:
            walls.add((r, c))
    for r, c in walls:
        grid[r, c] = -1.0

    coins = set()
    while len(coins) < n_coins:
        r, c = rng.randint(0, size), rng.randint(0, size)
        if (r, c) not in walls and (r, c) not in [(0, 0), (size-1, size-1)] and grid[r, c] == 0:
            coins.add((r, c))
    for r, c in coins:
        grid[r, c] = 1.0

    return grid


def score_direction(grid, pos, d):
    """Score a single direction. Negative = wall/something bad."""
    size = grid.shape[0]
    r, c = pos[0] + d[0], pos[1] + d[1]

    if not (0 <= r < size and 0 <= c < size):
        return -10.0  # out of bounds = wall

    cell = grid[r, c]
    if cell < 0:
        return -10.0  # wall
    if cell >= 10.0:
        return 10.0   # goal
    if cell == 1.0:
        return 5.0    # coin
    # Empty: encourage movement toward unexplored
    return -0.5  # small penalty for stepping on nothing


def run_grid_fixed(grid, tau=None, temperature=None, max_steps=300, seed=42):
    rng = np.random.RandomState(seed)
    size = grid.shape[0]
    pos = (0, 0)
    total_reward = 0.0
    walls_hit = 0
    choices_log = []
    scores_log = []

    for step in range(max_steps):
        # Score all 4 directions
        scores_4 = np.array([score_direction(grid, pos, d) for d in DIRS])

        # Record what the scores look like for analysis
        if step == 0 or step % 50 == 0:
            scores_log.append(scores_4.copy())

        # Handle case where agent is trapped (all directions bad)
        if np.all(scores_4 < 0):
            break

        chosen_i, _ = decide(scores_4.reshape(1, -1), tau=tau, temperature=temperature, rng=rng)
        chosen_i = int(chosen_i)
        choices_log.append(chosen_i)

        nr, nc = pos[0] + DIRS[chosen_i][0], pos[1] + DIRS[chosen_i][1]

        if not (0 <= nr < size and 0 <= nc < size):
            walls_hit += 1
            total_reward -= 0.5
            continue

        cell = grid[nr, nc]
        if cell < 0:  # wall
            walls_hit += 1
            total_reward -= 0.5
            continue

        pos = (nr, nc)

        if cell >= 10.0:  # goal
            total_reward += 10.0
            break
        elif cell == 1.0:  # coin
            total_reward += 1.0
            grid[nr, nc] = 0
        else:
            total_reward -= 0.05

    return {
        'steps': step + 1,
        'reward': total_reward,
        'walls_hit': walls_hit,
        'reached_goal': int(grid[pos] >= 10.0 if 0 <= pos[0] < size and 0 <= pos[1] < size else False),
        'scores_log': scores_log,  # for analysis
        'choices': choices_log,
    }


def run_grid_experiment(n_episodes=50):
    print("=" * 60)
    print("  E1: GRID (4-DIRECTION) — walls = -10, goal = +10, coin = +5")
    print("  Clamp test: wall directions are CLEARLY negative")
    print("=" * 60)

    tau_vals = [1.0, 3.0, 5.0, 10.0]
    temp_vals = [0.1, 0.3, 0.5, 1.0]

    stau = {}
    sm = {}

    print("  ── s^tau ──")
    for tau in tau_vals:
        steps_l, rew_l, walls_l, goal_l = [], [], [], []
        for ep in range(n_episodes):
            g = build_grid_fixed(seed=100 + ep)
            res = run_grid_fixed(g, tau=tau, seed=200 + ep)
            steps_l.append(res['steps'])
            rew_l.append(res['reward'])
            walls_l.append(res['walls_hit'])
            goal_l.append(res['reached_goal'])
        stau[tau] = {
            'avg_reward': np.mean(rew_l),
            'avg_walls': np.mean(walls_l),
            'goal_rate': np.mean(goal_l),
            'wall_action_pct': None,  # computed below
        }
        print(f"    tau={tau:<4.0f} | Reward={stau[tau]['avg_reward']:6.1f} "
              f"| Walls={stau[tau]['avg_walls']:5.1f} | Goal={stau[tau]['goal_rate']:.0%}")

    print("  ── Softmax ──")
    for temp in temp_vals:
        steps_l, rew_l, walls_l, goal_l = [], [], [], []
        for ep in range(n_episodes):
            g = build_grid_fixed(seed=100 + ep)
            res = run_grid_fixed(g, temperature=temp, seed=200 + ep)
            steps_l.append(res['steps'])
            rew_l.append(res['reward'])
            walls_l.append(res['walls_hit'])
            goal_l.append(res['reached_goal'])
        sm[temp] = {
            'avg_reward': np.mean(rew_l),
            'avg_walls': np.mean(walls_l),
            'goal_rate': np.mean(goal_l),
        }
        print(f"    T={temp:<5.2f} | Reward={sm[temp]['avg_reward']:6.1f} "
              f"| Walls={sm[temp]['avg_walls']:5.1f} | Goal={sm[temp]['goal_rate']:.0%}")

    # Wall-direction suppression test
    # Run one episode, measure: when 2+ directions are walls, does agent avoid them?
    g = build_grid_fixed(seed=999)
    for tau in tau_vals:
        res = run_grid_fixed(g, tau=tau, seed=999)
        wall_choices = sum(1 for c in res['choices']
                          if res['scores_log'][0] is not None)
        stau[tau]['wall_action_pct'] = compute_wall_avoidance(res, g)

    for temp in temp_vals:
        res = run_grid_fixed(g, temperature=temp, seed=999)
        sm[temp]['wall_action_pct'] = compute_wall_avoidance(res, g)

    best_s = max(stau, key=lambda t: stau[t]['avg_reward'])
    best_m = max(sm, key=lambda t: sm[t]['avg_reward'])
    print(f"\n  Best s^tau:  tau={best_s}  reward={stau[best_s]['avg_reward']:.1f}  "
          f"walls={stau[best_s]['avg_walls']:.1f}")
    print(f"  Best softmax: T={best_m}  reward={sm[best_m]['avg_reward']:.1f}  "
          f"walls={sm[best_m]['avg_walls']:.1f}")

    return stau, sm


def compute_wall_avoidance(res, grid):
    """Percentage of steps where agent picked a non-wall direction
    when wall directions were available."""
    if not res['choices']:
        return 0
    pos = (0, 0)
    wall_avoid = 0
    wall_opps = 0
    g = grid.copy()
    for step_i, chosen in enumerate(res['choices']):
        scores = np.array([score_direction(g, pos, d) for d in DIRS])
        wall_dirs = [i for i, s in enumerate(scores) if s <= -10.0]
        if wall_dirs:
            wall_opps += 1
            if chosen not in wall_dirs:
                wall_avoid += 1
        # Move
        pos = (pos[0] + DIRS[chosen][0], pos[1] + DIRS[chosen][1])
        if not (0 <= pos[0] < g.shape[0] and 0 <= pos[1] < g.shape[1]):
            pos = (max(0, min(g.shape[0]-1, pos[0])),
                   max(0, min(g.shape[1]-1, pos[1])))
        if g[pos] == 1.0:
            g[pos] = 0
    return wall_avoid / max(wall_opps, 1) * 100


# ═══════════════════════════════════════════════════
#  E2: CONTEXTUAL BANDIT (fixed: noise biased negative)
# ═══════════════════════════════════════════════════

def run_contextual_fixed(n_items=10, n_signal=10, n_noise=10,
                         n_rounds=1000, tau=None, temperature=None, seed=42):
    """
    10 items, 10 signal features (positive-biased), 10 noise features (negative-biased).
    True reward ONLY depends on signal features.
    Noise features are always harmful → clamp should zero them out.
    """
    rng = np.random.RandomState(seed)

    # Signal features: mean +0.3  (positively correlated with reward)
    # Noise features:  mean -0.3  (consistently drag scores down → negative)
    signal_feats = rng.randn(n_items, n_signal) * 0.3 + 0.3
    noise_feats = rng.randn(n_items, n_noise) * 0.3 - 0.3
    all_feats = np.hstack([signal_feats, noise_feats])

    # True weights: only signal matters
    true_w = np.zeros(n_signal + n_noise)
    true_w[:n_signal] = rng.randn(n_signal) * 0.5
    true_w[n_signal:] = 0.0  # noise features contribute ZERO to true reward

    # Online linear regression
    est_w = np.zeros(n_signal + n_noise)
    A_inv = np.eye(n_signal + n_noise) * 2.0

    pulls = np.zeros(n_items)
    items_rew = np.zeros(n_items)
    regrets = np.zeros(n_rounds)

    # Track attention
    attn_signal_frac = np.zeros(n_rounds)
    attn_noise_frac = np.zeros(n_rounds)

    for t in range(n_rounds):
        estimated = all_feats @ est_w

        if t > n_items:
            bonus = np.sqrt(2 * np.log(t + 1) / np.maximum(pulls, 1))
        else:
            bonus = np.ones(n_items) * 2.0

        scores = estimated + bonus

        chosen_i, _ = decide(scores.reshape(1, -1), tau=tau, temperature=temperature, rng=rng)
        chosen_i = int(chosen_i)

        # Reward from signal only
        true_mean = 1.0 / (1.0 + np.exp(-np.dot(signal_feats[chosen_i], true_w[:n_signal])))
        reward = float(rng.rand() < true_mean)

        pulls[chosen_i] += 1
        items_rew[chosen_i] += reward

        # Update regression
        x = all_feats[chosen_i]
        xxt = np.outer(x, x)
        A_inv -= (A_inv @ xxt @ A_inv) / (1 + x @ A_inv @ x)
        est_w = A_inv @ (all_feats.T @ (items_rew / np.maximum(pulls, 1).clip(0.01)))

        # True best
        all_true = 1.0 / (1.0 + np.exp(-(signal_feats @ true_w[:n_signal])))
        best_true = all_true.max()
        regrets[t] = best_true - all_true[chosen_i]

        # Feature attention: how much mass on signal vs noise?
        feat_scores = all_feats[chosen_i] * est_w
        if temperature is not None:
            attn = softmax_norm(feat_scores.reshape(1, -1), temperature).flatten()
        else:
            attn = stau_norm(feat_scores.reshape(1, -1), tau).flatten()
        attn_signal_frac[t] = attn[:n_signal].sum()
        attn_noise_frac[t] = attn[n_signal:].sum()

    return {
        'cumulative_regret': np.cumsum(regrets),
        'final_regret': np.cumsum(regrets)[-1],
        'signal_attention_avg': attn_signal_frac[300:].mean(),
        'noise_attention_avg': attn_noise_frac[300:].mean(),
    }


def run_contextual_experiment(n_runs=40):
    print()
    print("=" * 60)
    print("  E2: CONTEXTUAL (FIXED) — Signal=+0.3 bias, Noise=-0.3 bias")
    print("  Noise features consistently NEGATIVE → clamp should kill them")
    print("=" * 60)

    tau_vals = [1.0, 3.0, 5.0, 10.0, 20.0]
    temp_vals = [0.1, 0.3, 0.5, 1.0, 2.0]

    stau = {}
    sm = {}

    print("  ── s^tau ──")
    for tau in tau_vals:
        regrets = np.zeros((n_runs, 1000))
        sig_attns = np.zeros(n_runs)
        for run in range(n_runs):
            res = run_contextual_fixed(tau=tau, seed=42 + run)
            regrets[run] = res['cumulative_regret']
            sig_attns[run] = res['signal_attention_avg']
        stau[tau] = {
            'avg_regret': regrets.mean(axis=0),
            'final_regret': regrets.mean(axis=0)[-1],
            'signal_attn': sig_attns.mean(),
        }
        noise_pct = (1 - stau[tau]['signal_attn']) * 100
        print(f"    tau={tau:<5.0f} | Regret={stau[tau]['final_regret']:7.1f} "
              f"| Signal={stau[tau]['signal_attn']:.1%} | Noise={noise_pct:.1f}%")

    print("  ── Softmax ──")
    for temp in temp_vals:
        regrets = np.zeros((n_runs, 1000))
        sig_attns = np.zeros(n_runs)
        for run in range(n_runs):
            res = run_contextual_fixed(temperature=temp, seed=42 + run)
            regrets[run] = res['cumulative_regret']
            sig_attns[run] = res['signal_attention_avg']
        sm[temp] = {
            'avg_regret': regrets.mean(axis=0),
            'final_regret': regrets.mean(axis=0)[-1],
            'signal_attn': sig_attns.mean(),
        }
        noise_pct = (1 - sm[temp]['signal_attn']) * 100
        print(f"    T={temp:<5.2f}  | Regret={sm[temp]['final_regret']:7.1f} "
              f"| Signal={sm[temp]['signal_attn']:.1%} | Noise={noise_pct:.1f}%")

    best_s = min(stau, key=lambda t: stau[t]['final_regret'])
    best_m = min(sm, key=lambda t: sm[t]['final_regret'])
    print(f"\n  Best s^tau:  tau={best_s}  regret={stau[best_s]['final_regret']:.1f}  "
          f"signal focus={stau[best_s]['signal_attn']:.1%}")
    print(f"  Best softmax: T={best_m}  regret={sm[best_m]['final_regret']:.1f}  "
          f"signal focus={sm[best_m]['signal_attn']:.1%}")

    return stau, sm


# ═══════════════════════════════════════════════════
#  E3: POKER (fixed: high noise, real stakes)
# ═══════════════════════════════════════════════════

def run_poker_fixed(n_hands=800, tau=None, temperature=None, seed=42):
    """
    High noise (sigma=0.3). -EV actions are tempting because noise
    occasionally makes them look positive.

    Fold:  0
    Call:  2*s - 1  → negative when s < 0.5
    Raise: 3*s - 2  → negative when s < 0.67
    """
    rng = np.random.RandomState(seed)
    s = rng.rand(n_hands)

    evs = np.zeros((n_hands, 3))
    evs[:, 0] = 0.0
    evs[:, 1] = 2 * s - 1
    evs[:, 2] = 3 * s - 2

    optimal = evs.argmax(axis=1)
    optimal_ev = evs.max(axis=1)

    choices = np.zeros(n_hands, dtype=int)
    rewards = np.zeros(n_hands)
    neg_ev_picks = 0
    neg_ev_total = 0

    for h in range(n_hands):
        noise = rng.randn(3) * 0.3  # HIGH noise
        scores = evs[h] + noise

        chosen, _ = decide(scores.reshape(1, -1), tau=tau, temperature=temperature, rng=rng)
        chosen = int(chosen)
        choices[h] = chosen
        rewards[h] = evs[h, chosen]

        neg = [a for a in [1, 2] if evs[h, a] < -0.05]
        if neg:
            neg_ev_total += 1
            if chosen in neg:
                neg_ev_picks += 1

    regret = optimal_ev.sum() - rewards.sum()
    return {
        'regret': regret,
        'neg_ev_pct': neg_ev_picks / max(neg_ev_total, 1),
        'optimal_pct': (choices == optimal).mean(),
    }


def run_poker_experiment(n_runs=80):
    print()
    print("=" * 60)
    print("  E3: POKER (FIXED) — HIGH noise (0.3), real stakes")
    print("  Noise can make -EV look +EV → clamp resists temptation")
    print("=" * 60)

    tau_vals = [1.0, 2.0, 3.0, 5.0, 8.0, 15.0]
    temp_vals = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    stau = {}
    sm = {}

    print("  ── s^tau ──")
    for tau in tau_vals:
        regrets = []
        nevs = []
        opts = []
        for run in range(n_runs):
            res = run_poker_fixed(tau=tau, seed=42 + run)
            regrets.append(res['regret'])
            nevs.append(res['neg_ev_pct'])
            opts.append(res['optimal_pct'])
        stau[tau] = {
            'avg_regret': np.mean(regrets),
            'neg_ev_pct': np.mean(nevs),
            'optimal_pct': np.mean(opts),
        }
        print(f"    tau={tau:<5.0f} | Regret={stau[tau]['avg_regret']:7.2f} "
              f"| -EV%={stau[tau]['neg_ev_pct']:.1%} | Opt%={stau[tau]['optimal_pct']:.1%}")

    print("  ── Softmax ──")
    for temp in temp_vals:
        regrets = []
        nevs = []
        opts = []
        for run in range(n_runs):
            res = run_poker_fixed(temperature=temp, seed=42 + run)
            regrets.append(res['regret'])
            nevs.append(res['neg_ev_pct'])
            opts.append(res['optimal_pct'])
        sm[temp] = {
            'avg_regret': np.mean(regrets),
            'neg_ev_pct': np.mean(nevs),
            'optimal_pct': np.mean(opts),
        }
        print(f"    T={temp:<5.2f}  | Regret={sm[temp]['avg_regret']:7.2f} "
              f"| -EV%={sm[temp]['neg_ev_pct']:.1%} | Opt%={sm[temp]['optimal_pct']:.1%}")

    best_s = min(stau, key=lambda t: stau[t]['avg_regret'])
    best_m = min(sm, key=lambda t: sm[t]['avg_regret'])
    print(f"\n  Best s^tau:  tau={best_s}  regret={stau[best_s]['avg_regret']:.2f}  "
          f"-EV%={stau[best_s]['neg_ev_pct']:.1%}")
    print(f"  Best softmax: T={best_m}  regret={sm[best_m]['avg_regret']:.2f}  "
          f"-EV%={sm[best_m]['neg_ev_pct']:.1%}")

    return stau, sm


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    gs, gm = run_grid_experiment(n_episodes=50)
    cs, cm = run_contextual_experiment(n_runs=40)
    ps, pm = run_poker_experiment(n_runs=80)

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("s^tau vs Softmax: Three RL Games (With Real Negative Scores)",
                 fontsize=14, fontweight='bold')

    tl_g = sorted(gs.keys())
    tm_g = sorted(gm.keys())

    # [0,0] Grid: reward
    ax = axes[0, 0]
    ax.bar([0, 1], [gs[max(gs, key=lambda t: gs[t]['avg_reward'])]['avg_reward'],
                     gm[max(gm, key=lambda t: gm[t]['avg_reward'])]['avg_reward']],
           color=['#58a6ff', '#f78166'], width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f's^tau\n(best tau)', f'Softmax\n(best T)'], fontsize=9)
    ax.set_title('GRID: Avg Reward')
    ax.grid(True, alpha=0.3, axis='y')

    # [0,1] Grid: wall avoidance
    ax = axes[0, 1]
    wall_s = [gs[t].get('wall_action_pct', gs[t]['avg_walls']) for t in tl_g]
    wall_m = [gm[t].get('wall_action_pct', gm[t]['avg_walls']) for t in tm_g]
    ax.plot(tl_g, [gs[t]['avg_walls'] for t in tl_g], 'o-', color='#58a6ff', linewidth=2,
            markersize=8, label='s^tau (wall hits)')
    ax.plot(tm_g, [gm[t]['avg_walls'] for t in tm_g], 's--', color='#f78166', linewidth=2,
            markersize=8, label='Softmax (wall hits)')
    ax.set_title('GRID: Wall Hits per Episode (fewer=better)')
    ax.set_xlabel('Parameter'); ax.set_ylabel('Wall hits')
    ax.legend(); ax.grid(True, alpha=0.3)

    # [0,2] Grid: goal rate
    ax = axes[0, 2]
    ax.plot(tl_g, [gs[t]['goal_rate'] * 100 for t in tl_g], 'o-', color='#58a6ff',
            linewidth=2, markersize=8, label='s^tau')
    ax.plot(tm_g, [gm[t]['goal_rate'] * 100 for t in tm_g], 's--', color='#f78166',
            linewidth=2, markersize=8, label='Softmax')
    ax.set_title('GRID: Goal Rate')
    ax.set_xlabel('Parameter'); ax.set_ylabel('Goal %')
    ax.set_ylim(0, 105); ax.legend(); ax.grid(True, alpha=0.3)

    tl_c = sorted(cs.keys())
    tm_c = sorted(cm.keys())

    # [1,0] Contextual: signal attention
    ax = axes[1, 0]
    ax.plot(tl_c, [cs[t]['signal_attn'] * 100 for t in tl_c], 'o-', color='#58a6ff',
            linewidth=2, markersize=8, label='s^tau')
    ax.plot(tm_c, [cm[t]['signal_attn'] * 100 for t in tm_c], 's--', color='#f78166',
            linewidth=2, markersize=8, label='Softmax')
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5, label='50% (chance)')
    ax.set_title('CONTEXTUAL: Signal Feature Attention')
    ax.set_xlabel('Parameter'); ax.set_ylabel('Signal Attention %')
    ax.legend(); ax.grid(True, alpha=0.3)

    # [1,1] Contextual: regret
    ax = axes[1, 1]
    best_cs = min(cs, key=lambda t: cs[t]['final_regret'])
    best_cm = min(cm, key=lambda t: cm[t]['final_regret'])
    ax.plot(cs[best_cs]['avg_regret'], color='#58a6ff', linewidth=2.5,
            label=f's^tau tau={best_cs}')
    ax.plot(cm[best_cm]['avg_regret'], color='#f78166', linewidth=2.5,
            label=f'Softmax T={best_cm}')
    ax.set_title('CONTEXTUAL: Cumulative Regret')
    ax.set_xlabel('Round'); ax.set_ylabel('Regret')
    ax.legend(); ax.grid(True, alpha=0.3)

    tl_p = sorted(ps.keys())
    tm_p = sorted(pm.keys())

    # [1,2] Poker: -EV action rate
    ax = axes[1, 2]
    ax.plot(tl_p, [ps[t]['neg_ev_pct'] * 100 for t in tl_p], 'o-', color='#58a6ff',
            linewidth=2, markersize=8, label='s^tau')
    ax.plot(tm_p, [pm[t]['neg_ev_pct'] * 100 for t in tm_p], 's--', color='#f78166',
            linewidth=2, markersize=8, label='Softmax')
    ax.set_title('POKER: -EV Action Rate (lower=better)')
    ax.set_xlabel('Parameter'); ax.set_ylabel('-EV Action %')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    sp = Path(os.path.dirname(os.path.abspath(__file__))) / 'rl_games_fixed.png'
    plt.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure: {sp}")
