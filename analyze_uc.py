"""
Analyze UC sweep results: produce scaling plots + fitted exponents.

Reads uc_results.csv and produces:
  1. Total time vs n = N_g x T, by formulation
  2. Per-component breakdown: build / LP / MIP times
  3. LP relaxation gap vs problem size, by formulation
  4. Node count distribution
  5. Effect of horizon (T=24 vs T=72) on scaling
  6. Fitted scaling exponents per formulation
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
    'lines.markersize': 6,
})


def fit_powerlaw(x, y, x_min=None):
    x = np.asarray(x); y = np.asarray(y)
    valid = (y > 0)
    if x_min is not None:
        valid &= (x >= x_min)
    if valid.sum() < 2:
        return None, None, None
    logx = np.log(x[valid])
    logy = np.log(y[valid])
    beta, log_a = np.polyfit(logx, logy, 1)
    a = np.exp(log_a)
    ss_res = np.sum((logy - (beta * logx + log_a)) ** 2)
    ss_tot = np.sum((logy - np.mean(logy)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return beta, a, r2


def main(csv_path='uc_results.csv', out_dir='plots'):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df['converged'] == True].copy()
    df['n_x_T'] = df['n_gen'] * df['horizon']
    print(f"Loaded {len(df)} converged runs from {csv_path}")
    print(f"Sizes (N_g): {sorted(df['n_gen'].unique())}")
    print(f"Horizons: {sorted(df['horizon'].unique())}")
    print(f"Formulations: {sorted(df['formulation'].unique())}")

    # Aggregate replicates
    agg = df.groupby(['n_gen', 'horizon', 'formulation']).agg({
        'mip_total_time_s': ['mean', 'std', 'min'],
        'lp_relax_time_s': ['mean', 'std'],
        'build_time_s': 'mean',
        'mip_node_count': 'mean',
        'root_lp_gap_pct': 'mean',
        'n_var': 'mean',
        'n_constr': 'mean',
        'n_var_binary': 'mean',
        'lp_relax_iters': 'mean',
        'mip_simplex_iters': 'mean',
    }).reset_index()
    agg.columns = ['_'.join(c).rstrip('_') for c in agg.columns]
    agg['n_x_T'] = agg['n_gen'] * agg['horizon']

    colors = {'three_bin': 'C0', 'three_bin_tight': 'C1', 'perspective': 'C2'}
    markers = {'three_bin': 'o', 'three_bin_tight': 's', 'perspective': '^'}
    labels = {'three_bin': 'Three-bin (loose)', 'three_bin_tight': 'Three-bin (tight)',
              'perspective': 'Perspective (PWL=4)'}

    # ------------------------------------------------------------------
    # Plot 1: Total MIP time vs problem size n = N_g * T (T=24 only)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    fit_table = []
    sub = agg[agg['horizon'] == 24]

    for form in ['three_bin', 'three_bin_tight', 'perspective']:
        s = sub[sub['formulation'] == form].sort_values('n_x_T')
        if s.empty:
            continue
        ax.errorbar(s['n_x_T'], s['mip_total_time_s_mean'],
                    yerr=s['mip_total_time_s_std'].fillna(0),
                    marker=markers[form], color=colors[form], label=labels[form],
                    capsize=3)
        beta, a, r2 = fit_powerlaw(s['n_x_T'], s['mip_total_time_s_mean'])
        if beta is not None:
            fit_table.append({'formulation': form, 'horizon': 24,
                              'beta': beta, 'r2': r2, 'a': a})
            xfit = np.array([s['n_x_T'].min(), s['n_x_T'].max()])
            yfit = a * xfit ** beta
            ax.plot(xfit, yfit, '--', color=colors[form], alpha=0.5,
                    label=f'  fit: $n^{{{beta:.2f}}}$')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'Problem size $n = N_g \times T$')
    ax.set_ylabel('MIP total time [s]')
    ax.set_title('UC total solve time vs problem size (T=24)')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / '01_total_time_vs_n.png', dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: Per-component breakdown (three_bin, T=24)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    s = agg[(agg['formulation'] == 'three_bin') & (agg['horizon'] == 24)].sort_values('n_gen')

    components = ['build_time_s_mean', 'lp_relax_time_s_mean']
    labels_comp = ['Build (model construction)', 'LP relaxation']
    # Add an "MIP solve - LP" component
    s['mip_minus_lp'] = (s['mip_total_time_s_mean'] - s['lp_relax_time_s_mean']).clip(lower=1e-4)

    ax = axes[0]
    bottom = np.zeros(len(s))
    for comp, label in zip(components + ['mip_minus_lp'],
                            labels_comp + ['MIP B&C beyond root LP']):
        vals = s[comp].values
        ax.fill_between(s['n_gen'], bottom, bottom + vals, label=label, alpha=0.7)
        bottom += vals
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$N_g$')
    ax.set_ylabel('Time [s]')
    ax.set_title('Per-component time breakdown (three_bin, T=24)')
    ax.legend(loc='upper left', fontsize=8)

    ax = axes[1]
    total = s[components + ['mip_minus_lp']].sum(axis=1)
    for comp, label in zip(components + ['mip_minus_lp'],
                            labels_comp + ['MIP B&C beyond root LP']):
        pct = 100 * s[comp].values / total.values
        ax.plot(s['n_gen'], pct, marker='o', label=label)
    ax.set_xscale('log')
    ax.set_xlabel(r'$N_g$')
    ax.set_ylabel('% of total time')
    ax.set_title('Where the time goes')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim([0, 100])
    plt.tight_layout()
    plt.savefig(out_dir / '02_breakdown.png', dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: LP relaxation gap vs problem size
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    sub = agg[agg['horizon'] == 24]
    for form in ['three_bin', 'three_bin_tight', 'perspective']:
        s = sub[sub['formulation'] == form].sort_values('n_gen')
        if s.empty:
            continue
        ax.plot(s['n_gen'], s['root_lp_gap_pct_mean'],
                marker=markers[form], color=colors[form], label=labels[form])
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$N_g$')
    ax.set_ylabel('Root LP gap [%]')
    ax.set_title('LP relaxation tightness vs network size (T=24)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / '03_lp_gap.png', dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 4: Effect of horizon on time scaling
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    for T_val in [24, 72]:
        s = agg[(agg['formulation'] == 'three_bin') & (agg['horizon'] == T_val)].sort_values('n_gen')
        if s.empty:
            continue
        ax.errorbar(s['n_gen'], s['mip_total_time_s_mean'],
                    yerr=s['mip_total_time_s_std'].fillna(0),
                    marker='o' if T_val == 24 else 's',
                    label=f'T = {T_val}', capsize=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$N_g$')
    ax.set_ylabel('MIP total time [s]')
    ax.set_title('Effect of horizon T on solve time (three_bin)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / '04_horizon_effect.png', dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 5: Node count vs problem size (mostly 1 here, but informative)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    sub = agg[agg['horizon'] == 24]
    for form in ['three_bin', 'three_bin_tight', 'perspective']:
        s = sub[sub['formulation'] == form].sort_values('n_gen')
        if s.empty:
            continue
        ax.plot(s['n_gen'], s['mip_node_count_mean'],
                marker=markers[form], color=colors[form], label=labels[form])
    ax.set_xscale('log')
    ax.set_xlabel(r'$N_g$')
    ax.set_ylabel('B&C nodes (mean)')
    ax.set_title('Branch-and-bound node count vs network size (T=24)')
    ax.set_ylim([0, max(2, agg['mip_node_count_mean'].max() * 1.2)])
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / '05_node_count.png', dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Plot 6: Variable counts grow as expected
    # ------------------------------------------------------------------
    fig, ax = plt.subplots()
    sub = agg[agg['horizon'] == 24]
    for form in ['three_bin', 'three_bin_tight', 'perspective']:
        s = sub[sub['formulation'] == form].sort_values('n_gen')
        if s.empty:
            continue
        ax.plot(s['n_gen'], s['n_var_mean'], marker=markers[form],
                color=colors[form], label=f'{labels[form]}: vars')
        ax.plot(s['n_gen'], s['n_constr_mean'], marker=markers[form],
                color=colors[form], linestyle='--', alpha=0.5,
                label=f'{labels[form]}: constraints')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$N_g$')
    ax.set_ylabel('Count')
    ax.set_title('Model size: variables and constraints (T=24)')
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / '06_model_size.png', dpi=120)
    plt.close()

    # ------------------------------------------------------------------
    # Print fit table & summary
    # ------------------------------------------------------------------
    fit_df = pd.DataFrame(fit_table)
    print("\n=== Fitted scaling exponents (MIP total time vs n=N_g*T, T=24) ===")
    print(fit_df.to_string(index=False, float_format='%.4f'))
    fit_df.to_csv(out_dir / 'scaling_fits.csv', index=False)

    print("\n=== Summary table (three_bin, T=24) ===")
    s = agg[(agg['formulation'] == 'three_bin') & (agg['horizon'] == 24)].sort_values('n_gen')
    cols = ['n_gen', 'n_x_T', 'n_var_mean', 'n_constr_mean', 'n_var_binary_mean',
            'mip_total_time_s_mean', 'mip_node_count_mean',
            'root_lp_gap_pct_mean', 'lp_relax_iters_mean']
    print(s[cols].to_string(index=False, float_format='%.3f'))
    s[cols].to_csv(out_dir / 'summary_three_bin.csv', index=False)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == '__main__':
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/uc_experiment/uc_results.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else '/home/claude/uc_experiment/plots'
    main(csv, out)
