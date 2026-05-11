"""
Analyze the v2 results: how does branching scale with symmetry?

Produces:
  1. N_nodes vs symmetry_score, by plateau
  2. Time vs N_nodes (almost linear)
  3. LP gap vs symmetry/plateau
  4. Node count by (pattern, plateau) heatmap
  5. Time scaling with horizon at each pattern
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'figure.figsize': (8, 5),
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
})


def fit_powerlaw(x, y, x_min=None):
    x = np.asarray(x); y = np.asarray(y)
    valid = (y > 0) & (x > 0)
    if x_min is not None:
        valid &= (x >= x_min)
    if valid.sum() < 2:
        return None, None, None
    logx = np.log(x[valid]); logy = np.log(y[valid])
    beta, log_a = np.polyfit(logx, logy, 1)
    a = np.exp(log_a)
    ss_res = np.sum((logy - (beta * logx + log_a)) ** 2)
    ss_tot = np.sum((logy - np.mean(logy)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return beta, a, r2


def main(csv_path='uc_v2_results.csv', out_dir='plots_v2'):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df['converged'] == True].copy()
    print(f"Loaded {len(df)} converged runs")

    # ------------------------------------------------------------------
    # Plot 1: Node count vs symmetry score, separated by plateau
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    plateau_colors = {0.0: 'C3', 1.0: 'C0'}
    plateau_labels = {0.0: 'plateau=0.0 (perfectly identical units)',
                      1.0: 'plateau=1.0 (random within type)'}

    for plateau in [1.0, 0.0]:
        sub = df[df['plateau'] == plateau]
        ax.scatter(sub['symmetry_score'], sub['mip_node_count'],
                   c=plateau_colors[plateau], label=plateau_labels[plateau],
                   alpha=0.7, s=60)

    # Fit
    for plateau in [1.0, 0.0]:
        sub = df[df['plateau'] == plateau]
        beta, a, r2 = fit_powerlaw(sub['symmetry_score'].values,
                                    sub['mip_node_count'].values)
        if beta is not None:
            xfit = np.array([sub['symmetry_score'].min(),
                             sub['symmetry_score'].max()])
            yfit = a * xfit ** beta
            ax.plot(xfit, yfit, '--', color=plateau_colors[plateau], alpha=0.5,
                    label=f'  plateau={plateau}: $N_{{\\mathrm{{nodes}}}} \\sim s^{{{beta:.2f}}}$ (R²={r2:.2f})')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Symmetry score $s$')
    ax.set_ylabel(r'B&C nodes $N_{\mathrm{nodes}}$')
    ax.set_title('Branching grows with symmetry; identical units explode the tree')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / '01_nodes_vs_symmetry.png', dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Time vs nodes (proves time = c * nodes asymptotically)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    for plateau in [1.0, 0.0]:
        sub = df[df['plateau'] == plateau]
        ax.scatter(sub['mip_node_count'], sub['mip_total_time_s'],
                   c=plateau_colors[plateau], label=plateau_labels[plateau],
                   alpha=0.7, s=60)

    # Linear fit
    valid = (df['mip_node_count'] > 0)
    beta, a, r2 = fit_powerlaw(df.loc[valid, 'mip_node_count'].values,
                                df.loc[valid, 'mip_total_time_s'].values)
    if beta is not None:
        xfit = np.array([df['mip_node_count'].min(), df['mip_node_count'].max()])
        yfit = a * xfit ** beta
        ax.plot(xfit, yfit, ':k', alpha=0.5,
                label=f'fit: time $\\sim N_{{\\mathrm{{nodes}}}}^{{{beta:.2f}}}$ (R²={r2:.2f})')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'B&C nodes $N_{\mathrm{nodes}}$')
    ax.set_ylabel('MIP wall-clock [s]')
    ax.set_title('Wall-clock is essentially linear in node count')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / '02_time_vs_nodes.png', dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: Effect of plateau (=symmetry) on node count, separated by pattern
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    pattern_colors = {'diverse': 'C0', 'sym4': 'C1', 'sym8': 'C2'}
    bar_x = np.arange(len(df['fleet_pattern'].unique()))
    patterns = ['diverse', 'sym4', 'sym8']

    width = 0.35
    means_p1 = []
    means_p0 = []
    for pat in patterns:
        m1 = df[(df['fleet_pattern']==pat) & (df['plateau']==1.0)]['mip_node_count'].mean()
        m0 = df[(df['fleet_pattern']==pat) & (df['plateau']==0.0)]['mip_node_count'].mean()
        means_p1.append(m1); means_p0.append(m0)

    ax.bar(bar_x - width/2, means_p1, width, label='plateau=1.0 (random)',
           color='C0', alpha=0.8)
    ax.bar(bar_x + width/2, means_p0, width, label='plateau=0.0 (identical)',
           color='C3', alpha=0.8)
    ax.set_xticks(bar_x); ax.set_xticklabels(patterns)
    ax.set_yscale('log')
    ax.set_xlabel('Fleet pattern')
    ax.set_ylabel('Mean B&C nodes')
    ax.set_title('Identical-unit fleets explode branching at scale')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / '03_plateau_effect.png', dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 4: Node count vs horizon T, separated by pattern (sym8 plateau=0 only)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    for pat in patterns:
        for plateau in [0.0, 1.0]:
            sub = df[(df['fleet_pattern']==pat) & (df['plateau']==plateau)]
            agg = sub.groupby('horizon')['mip_node_count'].mean().reset_index()
            if len(agg) > 0:
                style = '-' if plateau == 0.0 else '--'
                ax.plot(agg['horizon'], agg['mip_node_count'],
                        marker='o', linestyle=style, color=pattern_colors[pat],
                        label=f'{pat}, p={plateau}', alpha=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('Horizon T')
    ax.set_ylabel('Mean B&C nodes')
    ax.set_title('Tree size grows with horizon; effect dominated by symmetry')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / '04_horizon_effect.png', dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 5: LP gap vs configuration
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    means_p1 = []
    means_p0 = []
    for pat in patterns:
        m1 = df[(df['fleet_pattern']==pat) & (df['plateau']==1.0)]['root_lp_gap_pct'].mean()
        m0 = df[(df['fleet_pattern']==pat) & (df['plateau']==0.0)]['root_lp_gap_pct'].mean()
        means_p1.append(abs(m1) if not np.isnan(m1) else 0)
        means_p0.append(abs(m0) if not np.isnan(m0) else 0)
    ax.bar(bar_x - width/2, means_p1, width, label='plateau=1.0', color='C0', alpha=0.8)
    ax.bar(bar_x + width/2, means_p0, width, label='plateau=0.0', color='C3', alpha=0.8)
    ax.set_xticks(bar_x); ax.set_xticklabels(patterns)
    ax.set_xlabel('Fleet pattern')
    ax.set_ylabel('Mean LP gap [%] (absolute)')
    ax.set_title('LP gap is small in all cases — it\'s the LP solution shape that drives branching, not the gap magnitude')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / '05_lp_gap.png', dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Summary tables
    # ------------------------------------------------------------------
    print("\n=== Summary: node count by (pattern, plateau) ===")
    summary = df.groupby(['fleet_pattern', 'plateau']).agg({
        'mip_node_count': ['mean', 'min', 'max'],
        'mip_total_time_s': ['mean', 'max'],
        'root_lp_gap_pct': 'mean',
    }).round(2)
    print(summary)
    summary.to_csv(out_dir / 'summary_by_config.csv')

    print("\n=== Symmetry score quartiles ===")
    df['sym_quartile'] = pd.qcut(df['symmetry_score'], q=4, duplicates='drop')
    sym_summary = df.groupby('sym_quartile').agg({
        'mip_node_count': 'mean',
        'mip_total_time_s': 'mean',
    }).round(1)
    print(sym_summary)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == '__main__':
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else 'uc_v2_results.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'plots_v2'
    main(csv, out)
