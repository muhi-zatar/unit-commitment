"""
V4 analysis: 2^4 factorial regression on the two instance sets.

Reads phase1_v4_results.csv (synth) and phase2_v4_results.csv (pglib).
Per instance set, produces:
  - pivot table: median nodes by feature combination
  - OLS regression: log10(nodes+1) ~ F1+F2+F3+F4 + 2-way interactions
  - main-effect bar charts (nodes, time, root_lp_gap)
  - half-normal plot of effect magnitudes

Plots written to plots/.
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from statsmodels.formula.api import ols
    HAS_SM = True
except ImportError:
    HAS_SM = False

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(THIS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

FEATURES = ['F1', 'F2', 'F3', 'F4']
FEATURE_NAMES = {
    'F1': 'PWL quadratic cost',
    'F2': 'Sep. startup/shutdown ramp',
    'F3': 'Lag-dependent startup',
    'F4': 'Must-run enforcement',
}


def _load(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None
    for c in ('mip_node_count', 'mip_total_time_s', 'root_lp_gap_pct',
              'F1', 'F2', 'F3', 'F4'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'status' in df.columns:
        df = df[df['status'].astype(str).str.lower().str.startswith('optimal')]
    return df.dropna(subset=FEATURES + ['mip_node_count'])


def pivot_by_features(df, value_col, agg='median'):
    return df.groupby(FEATURES)[value_col].agg(agg).reset_index()


def fit_regression(df, response='log_nodes'):
    if not HAS_SM:
        return None
    df = df.copy()
    df['log_nodes'] = np.log10(df['mip_node_count'].clip(lower=1) + 1)
    df['log_time']  = np.log10(df['mip_total_time_s'].clip(lower=1e-3))
    formula = (f"{response} ~ F1 + F2 + F3 + F4 + "
               f"F1:F2 + F1:F3 + F1:F4 + F2:F3 + F2:F4 + F3:F4")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            model = ols(formula, data=df).fit()
        except Exception as e:
            print(f"regression failed for {response}: {e}")
            return None
    return model


def main_effects(df, value_col):
    """Return Series of mean(value | F=1) - mean(value | F=0) per feature."""
    out = {}
    for f in FEATURES:
        m1 = df[df[f] == 1][value_col].median()
        m0 = df[df[f] == 0][value_col].median()
        out[f] = m1 - m0
    return pd.Series(out)


def plot_main_effects_bars(df, label):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title, log_y in zip(
            axes,
            ['mip_node_count', 'mip_total_time_s', 'root_lp_gap_pct'],
            ['MIP nodes', 'MIP wall time (s)', 'Root LP gap (%)'],
            [True, True, False]):
        med0 = [df[df[f] == 0][col].median() for f in FEATURES]
        med1 = [df[df[f] == 1][col].median() for f in FEATURES]
        x = np.arange(len(FEATURES))
        ax.bar(x - 0.18, med0, width=0.36, label='F=0', color='C0')
        ax.bar(x + 0.18, med1, width=0.36, label='F=1', color='C3')
        ax.set_xticks(x); ax.set_xticklabels(FEATURES)
        ax.set_title(title)
        if log_y:
            ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
    fig.suptitle(f'V4 main effects — {label}')
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f'v4_main_effects_{label}.png')
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


def plot_half_normal(model, label):
    if model is None:
        return
    coefs = model.params.drop('Intercept', errors='ignore')
    coefs = coefs.abs().sort_values()
    n = len(coefs)
    # half-normal quantiles
    p = (np.arange(1, n + 1) - 0.5) / n
    qq = np.sqrt(2) * np.abs(np.array([
        np.percentile(np.random.randn(10000), 100 * pp) for pp in (p + 1) / 2
    ]))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(qq, coefs.values, s=30)
    for x, y, name in zip(qq, coefs.values, coefs.index):
        ax.annotate(name, (x, y), fontsize=8, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('half-normal quantile')
    ax.set_ylabel(f'|effect on {model.model.endog_names}|')
    ax.set_title(f'V4 half-normal plot — {label}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, f'v4_halfnormal_{label}.png')
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


def analyze_one(df, label):
    print(f"\n=========================================")
    print(f"=== V4 analysis: {label}  (n={len(df)} rows)")
    print(f"=========================================")
    if len(df) == 0:
        print("(no rows)")
        return
    print("\nMain effects (median F=1 - median F=0):")
    for col, fmt in [('mip_node_count', 'nodes'),
                     ('mip_total_time_s', 'time(s)'),
                     ('root_lp_gap_pct', 'gap%')]:
        if col not in df.columns:
            continue
        eff = main_effects(df, col)
        print(f"  {fmt:8s}: " + ", ".join(f"{f}={v:+.3f}" for f, v in eff.items()))

    print(f"\nPivot table (median nodes by F1F2F3F4):")
    pivot = pivot_by_features(df, 'mip_node_count', 'median')
    pivot['label'] = pivot.apply(
        lambda r: ''.join(str(int(r[f])) for f in FEATURES), axis=1)
    print(pivot[['label', 'mip_node_count']].to_string(index=False))

    if HAS_SM:
        for response in ('log_nodes', 'log_time'):
            model = fit_regression(df, response=response)
            if model is None:
                continue
            print(f"\nOLS: {response}  R^2={model.rsquared:.3f}")
            tbl = pd.DataFrame({
                'coef': model.params,
                'std_err': model.bse,
                't': model.tvalues,
                'p>|t|': model.pvalues,
            })
            print(tbl.to_string(float_format=lambda x: f"{x:.3f}"))
            plot_half_normal(model, f"{label}_{response}")

    plot_main_effects_bars(df, label)


def acceptance_summary(df_synth, df_pglib):
    print("\n=== V4 acceptance criterion ===")
    print("(criterion: at least one feature swings median nodes >= 3x on one set)")
    for label, df in [('synth', df_synth), ('pglib', df_pglib)]:
        if df is None or len(df) == 0:
            continue
        for f in FEATURES:
            r1 = df[df[f] == 1]['mip_node_count'].median()
            r0 = df[df[f] == 0]['mip_node_count'].median()
            ratio = (max(r1, 1) / max(r0, 1)) if r0 > 0 else float('nan')
            print(f"  {label} {f}: median(F=1)/median(F=0) = "
                  f"{r1:.1f}/{r0:.1f} = {ratio:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synth-csv',
                        default=os.path.join(THIS_DIR, 'phase1_v4_results.csv'))
    parser.add_argument('--pglib-csv',
                        default=os.path.join(THIS_DIR, 'phase2_v4_results.csv'))
    args = parser.parse_args()

    df_synth = _load(args.synth_csv)
    df_pglib = _load(args.pglib_csv)

    if df_synth is not None:
        analyze_one(df_synth, 'synth')
    else:
        print(f"(no {args.synth_csv})")
    if df_pglib is not None:
        analyze_one(df_pglib, 'pglib')
    else:
        print(f"(no {args.pglib_csv})")

    acceptance_summary(df_synth, df_pglib)


if __name__ == '__main__':
    main()
