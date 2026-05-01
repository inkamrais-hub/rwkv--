"""
s^tau vs Softmax vs ε-greedy in Real Q-Learning

8x8 grid world. Agent learns via tabular Q-learning.
At each step: advantage = Q[s] - mean(Q[s]).
Policy = s^tau(advantage) | softmax(advantage) | ε-greedy.

The question: when advantage goes NEGATIVE (action is below average),
does s^tau's clamp help the agent learn faster by suppressing bad actions?
"""
import numpy as np
from pathlib import Path
import os, sys

EPS = 1e-8


def stau_probs(scores, tau):
    clamped = np.maximum(scores, EPS)
    pw = np.power(clamped, tau)
    s = pw.sum()
    return pw / s if s > 0 else np.ones(len(scores)) / len(scores)


def softmax_probs(scores, temp):
    s = scores - scores.max()
    ex = np.exp(s / max(temp, 0.01))
    return ex / ex.sum()


# ═══════════════════════════════════════════════════
#  Environment
# ═══════════════════════════════════════════════════

class GridWorld:
    def __init__(self, size=8, seed=42):
        self.size = size
        self.rng = np.random.RandomState(seed)

        # FIXED LAYOUT: traps on diagonal between start and goal.
        # Shortest path S→G crosses 4 traps. Safe path goes around.
        self.grid = np.zeros((size, size))
        self.grid[0, 0] = 0
        self.grid[-1, -1] = 10  # goal

        # Traps on diagonal — block the straight shot
        trap_cells = {(2, 2), (3, 3), (4, 4), (5, 5)}
        for r, c in trap_cells:
            self.grid[r, c] = -3

        # Coins off the diagonal — reward the long way around
        coin_cells = {(0, 3), (1, 5), (2, 6), (3, 1), (4, 1),
                      (5, 2), (5, 6), (6, 3), (6, 5), (7, 0)}
        for r, c in coin_cells:
            self.grid[r, c] = 1

        self.trap_cells = trap_cells
        self.coin_cells = coin_cells
        self.collected = set()

    def reset(self):
        self.collected = set()
        self.pos = (0, 0)
        return 0

    def state_idx(self):
        return self.pos[0] * self.size + self.pos[1]

    def step(self, action):
        dr = [-1, 1, 0, 0][action]
        dc = [0, 0, -1, 1][action]
        nr = max(0, min(self.size - 1, self.pos[0] + dr))
        nc = max(0, min(self.size - 1, self.pos[1] + dc))
        self.pos = (nr, nc)

        cell = self.grid[nr, nc]
        reward = -0.1

        if cell >= 10:
            reward += 10
            done = False  # DON'T end — agent keeps bouncing, Q converges honestly
        elif cell == 1 and (nr, nc) not in self.collected:
            reward += 2
            self.collected.add((nr, nc))
            done = False
        elif cell < 0:
            reward += cell  # -3
            done = False
        else:
            done = False

        return reward, done


# ═══════════════════════════════════════════════════
#  Agent with Q-learning
# ═══════════════════════════════════════════════════

def run_qlearning(env, n_episodes=500, lr=0.1, gamma=0.95,
                  policy='stau', tau=None, temp=None, epsilon=None,
                  seed=42):
    """
    Q-learning with advantage-based policy.

    policy types:
      'stau':   probs = s^tau( Q[s] - mean(Q[s]) )
      'softmax': probs = softmax( Q[s] - mean(Q[s]), temp )
      'egreedy': epsilon-greedy on argmax Q[s]
    """
    rng = np.random.RandomState(seed)
    n_states = env.size * env.size
    n_actions = 4

    # Zero initialization — Q for bad actions will go NEGATIVE
    Q = np.zeros((n_states, n_actions))

    episode_rewards = np.zeros(n_episodes)
    episode_steps = np.zeros(n_episodes, dtype=int)
    episode_goal = np.zeros(n_episodes, dtype=bool)
    episode_trap = np.zeros(n_episodes)
    trap_visits = np.zeros(n_episodes)

    # Track: % probability mass on NEGATIVE-Q actions (clamp kills these)
    avg_neg_q_mass = np.zeros(n_episodes)

    for ep in range(n_episodes):
        env.reset()
        total_reward = 0.0
        steps = 0
        visited_goal = False
        traps_this_ep = 0

        for _ in range(200):
            s = env.state_idx()
            qs = Q[s]

            if policy == 'stau':
                probs = stau_probs(qs, tau)
            elif policy == 'softmax':
                probs = softmax_probs(qs, temp)
            elif policy == 'egreedy':
                best = np.argmax(qs)
                probs = np.ones(n_actions) * epsilon / n_actions
                probs[best] += 1 - epsilon
            else:
                raise ValueError(policy)

            # Probability mass on negative-Q actions (clamp target)
            neg_mask = qs < -0.01
            if neg_mask.any():
                avg_neg_q_mass[ep] += probs[neg_mask].sum()

            # Sample action
            a = rng.choice(n_actions, p=probs)
            reward, done = env.step(a)

            if env.grid[env.pos] >= 10:
                visited_goal = True

            # Q-learning update
            sp = env.state_idx()
            Q[s, a] += lr * (reward + gamma * Q[sp].max() - Q[s, a])

            total_reward += reward
            steps += 1

            # Track trap visits (after ep 50 — learning phase)
            if ep >= 50 and reward < -2.0:
                traps_this_ep += 1

            if done:
                break

        episode_rewards[ep] = total_reward
        episode_steps[ep] = steps
        episode_goal[ep] = visited_goal
        trap_visits[ep] = traps_this_ep
        # Average over episode steps
        if steps > 0:
            avg_neg_q_mass[ep] /= steps

    return {
        'rewards': episode_rewards,
        'steps': episode_steps,
        'goals': episode_goal,
        'trap_visits': trap_visits,
        'neg_q_mass': avg_neg_q_mass,
        'Q': Q,
    }


# ═══════════════════════════════════════════════════
#  Experiment runner
# ═══════════════════════════════════════════════════

def run_all(n_runs=30, n_episodes=400):
    configs = []
    # s^tau
    for tau in [1.0, 2.0, 3.0, 5.0, 10.0]:
        configs.append(('stau', tau, None, None, f's^tau τ={tau:.0f}'))
    # softmax
    for temp in [0.1, 0.3, 0.5, 1.0, 2.0]:
        configs.append(('softmax', None, temp, None, f'SM T={temp:.1f}'))
    # ε-greedy
    for eps in [0.05, 0.1, 0.2, 0.3]:
        configs.append(('egreedy', None, None, eps, f'ε-greedy ε={eps:.2f}'))

    print("=" * 70)
    print("  Q-LEARNING GRID WORLD — s^tau vs Softmax vs ε-greedy")
    print("  8x8 grid, 8 coins(+1), 8 traps(-3), goal(+10)")
    print(f"  {n_episodes} episodes × {n_runs} runs per config")
    print("=" * 70)
    print()

    all_results = {}

    for pol, tau, temp, eps, label in configs:
        rewards_runs = np.zeros((n_runs, n_episodes))
        goals_runs = np.zeros((n_runs, n_episodes))
        traps_runs = np.zeros((n_runs, n_episodes))
        neg_runs = np.zeros((n_runs, n_episodes))

        for r in range(n_runs):
            env = GridWorld(size=8, seed=42 + r)
            res = run_qlearning(
                env, n_episodes=n_episodes,
                policy=pol, tau=tau, temp=temp, epsilon=eps,
                seed=100 + r
            )
            rewards_runs[r] = res['rewards']
            goals_runs[r] = res['goals']
            traps_runs[r] = res['trap_visits']
            neg_runs[r] = res['neg_q_mass']

        # Smooth rewards for display (running mean window=20)
        smooth = lambda x: np.convolve(x, np.ones(20)/20, mode='valid')

        avg_reward = rewards_runs.mean(axis=0)
        avg_goal = goals_runs.mean(axis=0)
        avg_trap = traps_runs.mean(axis=0)
        avg_neg = neg_runs.mean(axis=0)

        # Late-stage metrics (last 100 episodes)
        late_reward = rewards_runs[:, -100:].mean()
        late_goal = goals_runs[:, -100:].mean()
        late_trap = traps_runs[:, -100:].mean()
        late_neg = neg_runs[:, -100:].mean()

        # Convergence: first episode where smoothed reward > threshold
        threshold = 5.0
        conv_eps = []
        for r in range(n_runs):
            sr = smooth(rewards_runs[r])
            idx = np.where(sr >= threshold)[0]
            conv_eps.append(idx[0] if len(idx) > 0 else n_episodes)

        all_results[label] = {
            'avg_reward': avg_reward,
            'smooth_reward': smooth(avg_reward),
            'avg_goal': avg_goal,
            'avg_trap': avg_trap,
            'avg_neg_mass': avg_neg,
            'late_reward': late_reward,
            'late_goal': late_goal,
            'late_trap': late_trap,
            'late_neg_mass': late_neg,
            'conv_episodes': np.mean(conv_eps),
            'conv_std': np.std(conv_eps),
        }

        print(f"  {label:<18s} | FinalRew={late_reward:6.2f} | Goal={late_goal:.0%} "
              f"| Traps/ep={late_trap:5.2f} | NegQ%={late_neg:5.1%} "
              f"| Converge={np.mean(conv_eps):6.0f}±{np.std(conv_eps):.0f}ep")

    print()
    return all_results


# ═══════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════

def plot_results(results, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Q-Learning Grid World: s^tau vs Softmax vs ε-greedy",
                 fontsize=14, fontweight='bold')

    # Separate configs by type
    stau_keys = [k for k in results if 's^tau' in k]
    sm_keys = [k for k in results if 'SM' in k]
    eg_keys = [k for k in results if 'ε' in k]

    stau_taus = [float(k.split('τ=')[1]) for k in stau_keys]
    sm_temps = [float(k.split('T=')[1]) for k in sm_keys]
    eg_epss = [float(k.split('ε=')[1]) for k in eg_keys]

    blue_colors = plt.cm.Blues(np.linspace(0.35, 0.95, len(stau_keys)))
    orange_colors = plt.cm.Oranges(np.linspace(0.35, 0.95, len(sm_keys)))
    green_colors = plt.cm.Greens(np.linspace(0.35, 0.95, len(eg_keys)))

    # [0,0] Learning curves: s^tau
    ax = axes[0, 0]
    for k, c, tau in zip(stau_keys, blue_colors, stau_taus):
        ax.plot(results[k]['smooth_reward'], color=c, linewidth=2,
                label=f'τ={tau:.0f}')
    ax.set_title('s^tau: Smoothed Reward')
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # [0,1] Learning curves: softmax
    ax = axes[0, 1]
    for k, c, t in zip(sm_keys, orange_colors, sm_temps):
        ax.plot(results[k]['smooth_reward'], color=c, linewidth=2,
                label=f'T={t:.1f}')
    ax.set_title('Softmax: Smoothed Reward')
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # [0,2] Learning curves: ε-greedy
    ax = axes[0, 2]
    for k, c, e in zip(eg_keys, green_colors, eg_epss):
        ax.plot(results[k]['smooth_reward'], color=c, linewidth=2,
                label=f'ε={e:.2f}')
    ax.set_title('ε-greedy: Smoothed Reward')
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # [1,0] Best of each type — overlay
    ax = axes[1, 0]
    best_stau = max(stau_keys, key=lambda k: results[k]['late_reward'])
    best_sm = max(sm_keys, key=lambda k: results[k]['late_reward'])
    best_eg = max(eg_keys, key=lambda k: results[k]['late_reward'])
    ax.plot(results[best_stau]['smooth_reward'], color='#58a6ff', linewidth=3,
            label=f'{best_stau} (reward={results[best_stau]["late_reward"]:.1f})')
    ax.plot(results[best_sm]['smooth_reward'], color='#f78166', linewidth=3,
            label=f'{best_sm} (reward={results[best_sm]["late_reward"]:.1f})')
    ax.plot(results[best_eg]['smooth_reward'], color='#3fb950', linewidth=3,
            label=f'{best_eg} (reward={results[best_eg]["late_reward"]:.1f})')
    ax.set_title('Best of Each Type')
    ax.set_xlabel('Episode'); ax.set_ylabel('Reward')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # [1,1] Late-stage trap visits
    ax = axes[1, 1]
    all_late_trap = [results[k]['late_trap'] for k in stau_keys + sm_keys + eg_keys]
    all_labels = stau_keys + sm_keys + eg_keys
    all_colors = (['#58a6ff'] * len(stau_keys) +
                  ['#f78166'] * len(sm_keys) +
                  ['#3fb950'] * len(eg_keys))
    x = np.arange(len(all_labels))
    bars = ax.bar(x, all_late_trap, color=all_colors, alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=6.5, rotation=30)
    ax.set_title('Trap Visits per Episode (Late Stage, lower=better)')
    ax.set_ylabel('Traps/ep')
    ax.grid(True, alpha=0.3, axis='y')

    # [1,2] Negative-advantage mass vs final reward
    ax = axes[1, 2]
    for k in stau_keys:
        ax.scatter(results[k]['late_neg_mass'], results[k]['late_reward'],
                   color='#58a6ff', s=100, edgecolors='white', zorder=5)
        ax.annotate(f'τ={float(k.split("τ=")[1]):.0f}',
                    (results[k]['late_neg_mass'], results[k]['late_reward']),
                    fontsize=8, ha='left')
    for k in sm_keys:
        ax.scatter(results[k]['late_neg_mass'], results[k]['late_reward'],
                   color='#f78166', s=100, marker='s', edgecolors='white', zorder=5)
        ax.annotate(f'T={float(k.split("T=")[1]):.1f}',
                    (results[k]['late_neg_mass'], results[k]['late_reward']),
                    fontsize=8, ha='left')
    for k in eg_keys:
        ax.scatter(results[k]['late_neg_mass'], results[k]['late_reward'],
                   color='#3fb950', s=100, marker='^', edgecolors='white', zorder=5)
        ax.annotate(f'ε={float(k.split("ε=")[1]):.2f}',
                    (results[k]['late_neg_mass'], results[k]['late_reward']),
                    fontsize=8, ha='left')
    ax.set_title('Negative-Q Mass vs Final Reward')
    ax.set_xlabel('% Prob on Negative-Q Actions'); ax.set_ylabel('Final Reward')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {save_path}")


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    import sys, json
    OUT = Path(os.path.dirname(os.path.abspath(__file__)))

    results = run_all(n_runs=30, n_episodes=400)

    # Save results as JSON for easier reading
    json_out = {}
    for k, v in results.items():
        json_out[k] = {
            'late_reward': float(v['late_reward']),
            'late_goal': float(v['late_goal']),
            'late_trap': float(v['late_trap']),
            'late_neg_mass': float(v['late_neg_mass']),
            'conv_episodes': float(v['conv_episodes']),
        }
    with open(OUT / 'qlearn_results.json', 'w') as f:
        json.dump(json_out, f, indent=2)

    plot_results(results, OUT / 'qlearning_advantage.png')

    print("Results written to qlearn_results.json")
