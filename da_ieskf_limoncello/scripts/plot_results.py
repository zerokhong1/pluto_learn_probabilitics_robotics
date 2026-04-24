#!/usr/bin/env python3
"""
plot_results.py — generate all paper figures from batch evaluation results.

Produces:
  fig1_rmse_table.pdf       — ablation table (4 configs × all sequences)
  fig2_degenerate_bar.pdf   — degenerate sequences: pre/post LC RMSE grouped bar
  fig3_eigenvalues.pdf      — eigenvalue trace over time for one sequence
  fig4_pxx_ratio.pdf        — P[x,x] ratio DA/IE across sequences
  fig5_non_degen_control.pdf— control: non-degenerate sequences

Usage:
    python3 plot_results.py --results ~/results [--output ~/paper/figures]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi":      150,
    "text.usetex":     False,   # set True if LaTeX installed
})

COLORS = {
    "A": "#2171b5",   # IESKF no-LC     (blue)
    "B": "#ef3b2c",   # IESKF + LC      (red — this is the "bad" config)
    "C": "#41ab5d",   # DA-IESKF no-LC  (green)
    "D": "#984ea3",   # DA-IESKF + LC   (purple — the winner)
}
LABELS = {
    "A": "IESKF (no LC)",
    "B": "IESKF + LC",
    "C": "DA-IESKF (no LC)",
    "D": "DA-IESKF + LC",
}

DEGENERATE_SEQS    = ["city02", "ntu_day_01"]  # expected degenerate
NON_DEGENERATE_SEQS = ["ntu_day_02", "ntu_day_10", "grand_tour", "r_campus"]


# ── Load summary.csv ──────────────────────────────────────────────────────────

def load_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "summary.csv"
    if not path.exists():
        print(f"ERROR: summary.csv not found at {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    # Expected columns: config, sequence, rmse_m  (timestamp optional)
    required = {"config", "sequence", "rmse_m"}
    if not required.issubset(df.columns):
        # Try stripping timestamp column
        df.columns = [c.strip() for c in df.columns]
    return df


# ── Fig 1: RMSE ablation table ────────────────────────────────────────────────

def fig_rmse_table(df: pd.DataFrame, out_dir: Path):
    pivot = df.pivot_table(index="sequence", columns="config", values="rmse_m")
    pivot = pivot.reindex(columns=["A", "B", "C", "D"])

    fig, ax = plt.subplots(figsize=(6.5, max(2.0, 0.4 * len(pivot) + 1.0)))
    ax.axis("off")

    col_labels = [LABELS[c] for c in ["A", "B", "C", "D"]]
    cell_text  = []
    for _, row in pivot.iterrows():
        cell_text.append([
            f"{v:.3f}" if not np.isnan(v) else "—"
            for v in row
        ])

    table = ax.table(
        cellText=cell_text,
        rowLabels=list(pivot.index),
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Highlight best (lowest RMSE) per row in green, worst in red
    for i, (_, row) in enumerate(pivot.iterrows()):
        vals = row.dropna()
        if vals.empty: continue
        best_col  = vals.idxmin()
        worst_col = vals.idxmax()
        col_order = ["A", "B", "C", "D"]
        for j, cfg in enumerate(col_order):
            cell = table[(i + 1, j)]
            if cfg == best_col:
                cell.set_facecolor("#c7e9c0")
            elif cfg == worst_col:
                cell.set_facecolor("#fcbba1")

    ax.set_title("APE RMSE [m] — all sequences, all configs", pad=8)
    fig.tight_layout()
    path = out_dir / "fig1_rmse_table.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Fig 2: Degenerate sequences grouped bar ───────────────────────────────────

def fig_degenerate_bar(df: pd.DataFrame, out_dir: Path):
    degen_df = df[df["sequence"].isin(DEGENERATE_SEQS)]
    if degen_df.empty:
        print("No degenerate sequences in results — skipping fig2.")
        return

    seqs   = sorted(degen_df["sequence"].unique())
    n_seqs = len(seqs)
    cfgs   = ["A", "B", "C", "D"]
    x      = np.arange(n_seqs)
    width  = 0.18

    fig, ax = plt.subplots(figsize=(max(4.0, 1.8 * n_seqs), 3.5))
    for i, cfg in enumerate(cfgs):
        rmses = []
        for seq in seqs:
            row = degen_df[(degen_df["config"] == cfg) & (degen_df["sequence"] == seq)]
            rmses.append(row["rmse_m"].values[0] if not row.empty else np.nan)
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, rmses, width,
                      label=LABELS[cfg], color=COLORS[cfg], alpha=0.85,
                      edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Sequence")
    ax.set_ylabel("APE RMSE [m]")
    ax.set_title("Degenerate sequences: loop closure is harmful for IESKF,\nbut beneficial for DA-IESKF")
    ax.set_xticks(x)
    ax.set_xticklabels(seqs)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "fig2_degenerate_bar.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Fig 3: Eigenvalue trace ────────────────────────────────────────────────────

def fig_eigenvalue_trace(results_dir: Path, out_dir: Path,
                          seq: str = "city02", cfg: str = "D"):
    csv_path = results_dir / cfg / seq / "eigenvalues.csv"
    if not csv_path.exists():
        print(f"Eigenvalue log not found: {csv_path} — skipping fig3.")
        return

    eig_df = pd.read_csv(csv_path)
    if "timestamp" not in eig_df.columns:
        print("Eigenvalue CSV missing 'timestamp' column — skipping fig3.")
        return

    t = eig_df["timestamp"].values
    t = t - t[0]  # relative time

    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.0), sharex=True)

    ax = axes[0]
    if "lambda1" in eig_df.columns:
        ax.semilogy(t, eig_df["lambda1"].values, color=COLORS["A"], lw=0.8,
                    label=r"$\lambda_\mathrm{min}$")
    if "lambda3" in eig_df.columns:
        ax.semilogy(t, eig_df["lambda3"].values, color=COLORS["D"], lw=0.8,
                    label=r"$\lambda_\mathrm{max}$")
    ax.set_ylabel("Eigenvalue (log)")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    if "ratio" in eig_df.columns:
        ax2.plot(t, eig_df["ratio"].values, color=COLORS["C"], lw=0.8)
        ax2.axhline(0.15, color="red", ls="--", lw=0.8, label="Threshold 0.15")
        if "is_degenerate" in eig_df.columns:
            degen_t = t[eig_df["is_degenerate"].astype(bool)]
            ax2.scatter(degen_t, np.zeros_like(degen_t),
                        color="red", s=4, zorder=5, label="Degenerate")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel(r"$\lambda_\mathrm{min} / \lambda_\mathrm{max}$")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(framealpha=0.9)
    ax2.grid(alpha=0.3)

    axes[0].set_title(f"Eigenvalue analysis — {seq} (config {cfg})")
    fig.tight_layout()
    path = out_dir / "fig3_eigenvalues.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Fig 4: P[x,x] ratio ───────────────────────────────────────────────────────

def fig_pxx_ratio(df: pd.DataFrame, out_dir: Path):
    """
    Show ratio of RMSE improvement: (RMSE_A - RMSE_D) / RMSE_A  (positive = better).
    This is a proxy for the benefit of DA-IESKF + LC over baseline.
    """
    seqs = sorted(df["sequence"].unique())
    improvements = []
    for seq in seqs:
        rmse_A = df[(df["config"] == "A") & (df["sequence"] == seq)]["rmse_m"]
        rmse_D = df[(df["config"] == "D") & (df["sequence"] == seq)]["rmse_m"]
        if rmse_A.empty or rmse_D.empty:
            improvements.append(np.nan)
        else:
            imp = (rmse_A.values[0] - rmse_D.values[0]) / rmse_A.values[0] * 100
            improvements.append(imp)

    x = np.arange(len(seqs))
    colors = ["#41ab5d" if v > 0 else "#ef3b2c" for v in improvements]

    fig, ax = plt.subplots(figsize=(max(4.0, 1.4 * len(seqs)), 3.0))
    bars = ax.bar(x, improvements, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(seqs, rotation=20, ha="right")
    ax.set_ylabel("RMSE improvement [%]\n(A→D: baseline → DA-IESKF+LC)")
    ax.set_title("Benefit of DA-IESKF + Loop Closure vs. baseline IESKF\n(green = improvement, red = degradation)")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, improvements):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (1 if val >= 0 else -3),
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    path = out_dir / "fig4_improvement_pct.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Fig 5: Non-degenerate control ─────────────────────────────────────────────

def fig_non_degen_control(df: pd.DataFrame, out_dir: Path):
    non_degen_df = df[df["sequence"].isin(NON_DEGENERATE_SEQS)]
    if non_degen_df.empty:
        print("No non-degenerate sequences in results — skipping fig5.")
        return

    # Both IESKF and DA-IESKF should improve with LC (no harmful degradation)
    # Show this as two side-by-side groups: IESKF (A vs B), DA-IESKF (C vs D)
    seqs   = sorted(non_degen_df["sequence"].unique())
    x      = np.arange(len(seqs))
    width  = 0.2

    fig, ax = plt.subplots(figsize=(max(4.0, 1.8 * len(seqs)), 3.5))
    for i, cfg in enumerate(["A", "B", "C", "D"]):
        rmses = []
        for seq in seqs:
            row = non_degen_df[(non_degen_df["config"] == cfg) &
                               (non_degen_df["sequence"] == seq)]
            rmses.append(row["rmse_m"].values[0] if not row.empty else np.nan)
        offset = (i - 1.5) * width
        ax.bar(x + offset, rmses, width,
               label=LABELS[cfg], color=COLORS[cfg], alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Sequence")
    ax.set_ylabel("APE RMSE [m]")
    ax.set_title("Non-degenerate control: both filters benefit from LC\n(validates LC mechanism is correct)")
    ax.set_xticks(x)
    ax.set_xticklabels(seqs)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "fig5_non_degen_control.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path,
                        default=Path.home() / "results",
                        help="Path to results directory (default: ~/results)")
    parser.add_argument("--output", type=Path,
                        default=None,
                        help="Output directory for figures (default: results/figures)")
    args = parser.parse_args()

    results_dir = args.results
    out_dir     = args.output or results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    print(f"Saving figures to:    {out_dir}")
    print("")

    df = load_summary(results_dir)
    print(f"Loaded {len(df)} result rows.")
    print(df.to_string(index=False))
    print("")

    fig_rmse_table(df, out_dir)
    fig_degenerate_bar(df, out_dir)
    fig_eigenvalue_trace(results_dir, out_dir)
    fig_pxx_ratio(df, out_dir)
    fig_non_degen_control(df, out_dir)

    print("")
    print("=== All figures generated ===")
    for f in sorted(out_dir.glob("fig*.pdf")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
