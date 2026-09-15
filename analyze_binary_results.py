"""Validate METRA-FL experiment outputs and generate paper figures/tables.

Usage
-----
python analyze_binary_results.py --results results --out .
"""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


SEEDS = (11, 29, 47)
METHOD_ORDER = ["fedavg", "fedprox", "median", "trimmed_mean", "krum", "trust_v2x"]
LABELS = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "median": "Median",
    "trimmed_mean": "Trimmed mean",
    "krum": "Distance selector",
    "trust_v2x": "METRA-FL",
}


def validate(fed: pd.DataFrame, zero: pd.DataFrame) -> None:
    required_fed = {
        "seed", "study", "alpha", "clients", "aggregator", "malicious_fraction",
        "attack", "accuracy", "macro_f1", "mcc", "auroc", "fpr",
    }
    required_zero = {
        "seed", "held_out", "unknown_auroc", "unknown_aupr", "unknown_f1",
        "fpr95", "zero_day_fpr", "zero_day_recall", "zero_day_precision",
    }
    if missing := required_fed - set(fed):
        raise ValueError(f"Missing federated columns: {sorted(missing)}")
    if missing := required_zero - set(zero):
        raise ValueError(f"Missing held-out-family columns: {sorted(missing)}")
    if len(fed) != 78 or len(zero) != 27:
        raise ValueError(f"Expected 78 federated and 27 held-out-family rows, got {len(fed)} and {len(zero)}")
    if set(fed.seed) != set(SEEDS) or set(zero.seed) != set(SEEDS):
        raise ValueError("Seed set is incomplete")
    if fed.duplicated(["seed", "alpha", "clients", "aggregator", "malicious_fraction", "attack"]).any():
        raise ValueError("Duplicate federated experiment keys")
    if zero.duplicated(["seed", "held_out"]).any():
        raise ValueError("Duplicate held-out-family experiment keys")


def paired_tests(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot = base.pivot(index="seed", columns="aggregator", values="macro_f1")
    for method in METHOD_ORDER[:-1]:
        diff = pivot["trust_v2x"] - pivot[method]
        _, pvalue = ttest_rel(pivot["trust_v2x"], pivot[method])
        sd = diff.std(ddof=1)
        rows.append({
            "comparison": f"METRA-FL - {LABELS[method]}",
            "mean_difference": diff.mean(),
            "p_value_two_sided": pvalue,
            "cohens_dz": diff.mean() / sd if sd > 0 else np.nan,
            "n": len(diff),
        })
    return pd.DataFrame(rows)


def architecture_figure(path: Path) -> None:
    generator = Path(__file__).resolve().parent / "figures" / "generate_architecture.py"
    namespace = runpy.run_path(str(generator))
    namespace["main"](path, path.with_suffix(".png"))


def baseline_robustness_figure(fed: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.55))
    base = fed.query("study == 'baseline'")
    g = base.groupby("aggregator").macro_f1.agg(["mean", "std"]).reindex(METHOD_ORDER)
    x = np.arange(len(g)); ci = 4.302652729 * g["std"].to_numpy() / np.sqrt(3)
    colors = ["#7A9CC6"] * 5 + ["#C44E52"]
    axes[0].bar(x, 100*g["mean"], yerr=100*ci, capsize=3, color=colors, edgecolor="white")
    axes[0].set_xticks(x, [LABELS[m] for m in g.index], rotation=28, ha="right")
    axes[0].set_ylabel("Macro-F1 (%)"); axes[0].set_title("Clean, 20 clients, $\\alpha=0.3$")
    axes[0].grid(axis="y", alpha=.25); axes[0].set_ylim(70, 94)

    robust = fed.query("study == 'robustness'")
    methods = ["fedavg", "median", "krum", "trust_v2x"]
    attacks = ["label_flip", "sign_flip", "backdoor"]
    width = .19; x = np.arange(len(attacks))
    for j, method in enumerate(methods):
        vals = [100*robust.query("attack == @a and aggregator == @method").macro_f1.mean() for a in attacks]
        axes[1].bar(x + (j-1.5)*width, vals, width, label=LABELS[method])
    axes[1].set_xticks(x, ["Label flip", "Sign flip", "Backdoor"])
    axes[1].set_ylabel("Macro-F1 (%)"); axes[1].set_title("20% malicious clients")
    axes[1].grid(axis="y", alpha=.25); axes[1].set_ylim(35, 92)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def heterogeneity_figure(fed: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.35))
    base = fed.query("study == 'baseline' and aggregator in ['fedavg','trust_v2x']")
    noniid = pd.concat([base, fed.query("study == 'non_iid'")], ignore_index=True)
    scale = pd.concat([base, fed.query("study == 'scalability'")], ignore_index=True)
    for method, marker, color in [("fedavg", "o", "#4C72B0"), ("trust_v2x", "s", "#C44E52")]:
        q = noniid.query("aggregator == @method").groupby("alpha").macro_f1.mean().sort_index()
        axes[0].plot(q.index, 100*q.values, marker=marker, color=color, lw=1.8, label=LABELS[method])
        q = scale.query("aggregator == @method").groupby("clients").macro_f1.mean().sort_index()
        axes[1].plot(q.index, 100*q.values, marker=marker, color=color, lw=1.8, label=LABELS[method])
    axes[0].set_xlabel("Dirichlet concentration $\\alpha$"); axes[0].set_ylabel("Macro-F1 (%)")
    axes[0].set_title("Statistical heterogeneity (20 clients)")
    axes[1].set_xlabel("Number of clients"); axes[1].set_ylabel("Macro-F1 (%)")
    axes[1].set_title("Client-scale sensitivity ($\\alpha=0.3$)")
    for ax in axes:
        ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def zero_day_figure(zero: pd.DataFrame, path: Path) -> None:
    order = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms"]
    g = zero.groupby("held_out")[["unknown_auroc", "unknown_aupr"]].mean().reindex(order)
    x = np.arange(len(order)); width = .38
    fig, ax = plt.subplots(figsize=(10.5, 3.25))
    ax.bar(x-width/2, g.unknown_auroc, width, label="AUROC", color="#4C72B0")
    ax.bar(x+width/2, g.unknown_aupr, width, label="AUPR", color="#55A868")
    ax.set_xticks(x, order, rotation=28, ha="right"); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score"); ax.set_title("Post-preprocessing held-out-family transfer")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figures = args.out / "figures"; figures.mkdir(exist_ok=True)
    fed = pd.read_csv(args.results / "federated_raw.csv")
    zero = pd.read_csv(args.results / "zero_day_raw.csv")
    validate(fed, zero)
    base = fed.query("study == 'baseline'")
    tests = paired_tests(base); tests.to_csv(args.out / "paired_tests.csv", index=False)
    architecture_figure(figures / "architecture.pdf")
    baseline_robustness_figure(fed, figures / "baseline_robustness.pdf")
    heterogeneity_figure(fed, figures / "heterogeneity_scalability.pdf")
    zero_day_figure(zero, figures / "zero_day.pdf")
    summary = {
        "federated_runs": len(fed), "zero_day_runs": len(zero), "seeds": list(SEEDS),
        "clean": base.groupby("aggregator")[["accuracy", "macro_f1", "mcc", "auroc", "fpr"]].mean().to_dict("index"),
        "zero_day_macro": zero[["unknown_auroc", "unknown_aupr", "unknown_f1", "fpr95", "zero_day_recall", "zero_day_precision"]].mean().to_dict(),
    }
    (args.out / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print("Validated 78 federated and 27 held-out-family runs; generated four paper figures.")


if __name__ == "__main__":
    main()
