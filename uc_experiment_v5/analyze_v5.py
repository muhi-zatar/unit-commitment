"""
V5 analysis: log-log scaling of {nodes, time} vs network size.

Reads phase1_v5_results.csv. Writes:
  plots/v5_time_vs_size.png
  plots/v5_nodes_vs_size.png
  plots/v5_solver_compare.png (if both solvers present)
Prints scaling exponents and a per-network table.

All figures carry an "includes Pyomo build/IO overhead" annotation.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(THIS_DIR)
V3_PHASE5_CSV = os.path.join(PARENT, 'uc_experiment_v3', 'phase5_results.csv')
PLOTS_DIR = os.path.join(THIS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

OVERHEAD_NOTE = 'timings include Pyomo build + MPS write overhead'


def _load(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None


def _numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def _power_law_fit(x, y):
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 3:
        return None
    lx = np.log10(x[mask]); ly = np.log10(y[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    return slope, intercept


def plot_metric_vs_size(df, ycol, ylabel, fname, log_y=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    fits = {}
    for solver, grp in df.groupby('solver'):
        x = grp['n_branch'].values.astype(float)
        y = grp[ycol].values.astype(float)
        ax.scatter(x, y, s=40, alpha=0.8, label=solver)
        fit = _power_law_fit(x, y)
        if fit is not None:
            slope, intercept = fit
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, 10**(intercept) * xs**slope, '--', alpha=0.6,
                    label=f'{solver} fit: ~ n_branch^{slope:.2f}')
            fits[solver] = (slope, intercept)
    ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.set_xlabel('n_branch (transmission size)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    ax.set_title(f'V5 SCUC scaling — {OVERHEAD_NOTE}')
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, fname)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")
    return fits


def plot_solver_compare(df):
    """Bar chart: time and nodes per network, side-by-side HiGHS vs SCIP."""
    pivot_t = df.pivot_table(index='network', columns='solver',
                              values='mip_total_time_s', aggfunc='median')
    pivot_n = df.pivot_table(index='network', columns='solver',
                              values='mip_node_count', aggfunc='median')
    if pivot_t.shape[1] < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pivot_t.plot(kind='bar', ax=axes[0], logy=True, color=['C0', 'C3'])
    axes[0].set_ylabel('MIP wall time (s)')
    axes[0].set_title('HiGHS vs SCIP — solve time')
    axes[0].grid(True, alpha=0.3, axis='y', which='both')
    pivot_n.plot(kind='bar', ax=axes[1], logy=True, color=['C0', 'C3'])
    axes[1].set_ylabel('MIP nodes')
    axes[1].set_title('HiGHS vs SCIP — node count')
    axes[1].grid(True, alpha=0.3, axis='y', which='both')
    fig.suptitle(OVERHEAD_NOTE, fontsize=9)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, 'v5_solver_compare.png')
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',
                        default=os.path.join(THIS_DIR, 'phase1_v5_results.csv'))
    args = parser.parse_args()

    df = _load(args.csv)
    if df is None or len(df) == 0:
        print(f"no data at {args.csv}")
        sys.exit(0)

    df = _numeric(df, ['n_branch', 'n_bus', 'n_gen', 'horizon',
                       'mip_total_time_s', 'mip_node_count', 'mip_gap'])
    # Keep solver-converged or time-limited rows. SCIP reports 'gaplimit'
    # when it stopped within the requested gap; treat as success.
    if 'status' in df.columns:
        ok_prefixes = ('optimal', 'time', 'gaplimit')
        df = df[df['status'].astype(str).str.lower().str.startswith(ok_prefixes)]

    print(f"\nLoaded {len(df)} rows. solvers={list(df['solver'].unique())}")
    print("\n=== Per-network summary ===")
    summary = df.groupby(['network', 'solver']).agg(
        n_bus=('n_bus', 'first'),
        n_branch=('n_branch', 'first'),
        n_gen=('n_gen', 'first'),
        horizon=('horizon', 'first'),
        med_time=('mip_total_time_s', 'median'),
        med_nodes=('mip_node_count', 'median'),
        med_gap=('mip_gap', 'median'),
    ).reset_index()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Scaling fits ===")
    fits_time = plot_metric_vs_size(df, 'mip_total_time_s',
                                    'MIP wall time (s)',
                                    'v5_time_vs_size.png', log_y=True)
    fits_nodes = plot_metric_vs_size(df, 'mip_node_count',
                                     'MIP nodes',
                                     'v5_nodes_vs_size.png', log_y=True)
    if fits_time:
        for solver, (slope, _) in fits_time.items():
            print(f"  {solver:5s}: time ~ n_branch^{slope:.2f}")
    if fits_nodes:
        for solver, (slope, _) in fits_nodes.items():
            print(f"  {solver:5s}: nodes ~ n_branch^{slope:.2f}")

    if len(df['solver'].unique()) >= 2:
        plot_solver_compare(df)

    # V3 (PGLib single-bus) vs V5 (Egret SCUC with transmission) delta
    transmission_delta(df)


def transmission_delta(df_v5):
    """Compare V5 rts_gmlc rows (with transmission) to V3 phase5 rts_gmlc/*
    rows (same generators, single-bus). Prints a table; saves a bar plot."""
    if not os.path.exists(V3_PHASE5_CSV):
        print(f"\n(no V3 phase5 CSV at {V3_PHASE5_CSV} - "
              f"skipping transmission delta)")
        return
    try:
        df_v3 = pd.read_csv(V3_PHASE5_CSV)
    except Exception as e:
        print(f"could not read V3 CSV: {e}")
        return
    _numeric(df_v3, ['mip_total_time_s', 'mip_node_count', 'mip_gap', 'n_gen'])

    # V3 phase5 uses dataset='rts_gmlc' for PGLib's rts_gmlc/*.json cases
    v3_rts = df_v3[df_v3.get('dataset', '') == 'rts_gmlc'].copy()
    v3_rts = v3_rts[v3_rts['status'].astype(str).str.lower().str.startswith(
        ('optimal', 'time', 'gaplimit'))]
    v5_rts = df_v5[df_v5['network'] == 'rts_gmlc'].copy()

    if len(v3_rts) == 0 or len(v5_rts) == 0:
        print("\n(no comparable rts_gmlc rows in both CSVs)")
        return

    print("\n=== Transmission delta: V3 PGLib single-bus  vs  V5 Egret SCUC ===")
    print("(same RTS-GMLC generators on both sides; "
          "only the network model differs)")
    rows = []
    for solver in sorted(set(v3_rts['solver']) | set(v5_rts['solver'])):
        v3 = v3_rts[v3_rts['solver'] == solver]
        v5 = v5_rts[v5_rts['solver'] == solver]
        if len(v3) == 0 or len(v5) == 0:
            continue
        row = {
            'solver': solver,
            'v3_n_instances': len(v3),
            'v5_n_instances': len(v5),
            'v3_med_nodes': v3['mip_node_count'].median(),
            'v5_med_nodes': v5['mip_node_count'].median(),
            'nodes_x':      (v5['mip_node_count'].median()
                             / max(v3['mip_node_count'].median(), 1)),
            'v3_med_time':  v3['mip_total_time_s'].median(),
            'v5_med_time':  v5['mip_total_time_s'].median(),
            'time_x':       (v5['mip_total_time_s'].median()
                             / max(v3['mip_total_time_s'].median(), 1e-3)),
        }
        rows.append(row)
    if not rows:
        return
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda x: f"{x:.2f}"))

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    solvers = [r['solver'] for r in rows]
    x = np.arange(len(solvers))
    for ax, metric, ylabel in [
        (axes[0], 'nodes', 'MIP nodes (median)'),
        (axes[1], 'time',  'MIP wall time s (median)')]:
        v3_vals = [r[f'v3_med_{metric}'] for r in rows]
        v5_vals = [r[f'v5_med_{metric}'] for r in rows]
        ax.bar(x - 0.18, np.maximum(v3_vals, 1e-2), width=0.36,
               label='V3 single-bus PGLib', color='C0')
        ax.bar(x + 0.18, np.maximum(v5_vals, 1e-2), width=0.36,
               label='V5 with transmission', color='C3')
        ax.set_xticks(x); ax.set_xticklabels(solvers)
        ax.set_yscale('log')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y', which='both')
        ax.legend()
    fig.suptitle('Transmission delta — same RTS-GMLC generators, with vs without DC network',
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, 'v5_vs_v3_transmission_delta.png')
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
