#!/usr/bin/env python3
"""
No-task CF perplexity shift: ΔPPL = PPL_cf - PPL_or

Aggregation:
1) merge CF with OR on (subject_id, hadm_id)
2) dppl = ppl_cf - ppl_or
3) unit = per (subject_id, hadm_id, shift_bin): mean(dppl) across CF samples
4) plot mean(unit) per shift_bin with 95% CI (mean ± 1.96*SE)

Also outputs:
- ppl_shift_summary_by_shift.csv
- ppl_shift_overall_by_model.csv
- plot_avg_dppl_by_shift_notask_binned_extremes.png
- plot_avg_dppl_by_shift_notask_binned_extremes.pdf

IMPORTANT:
- Only models explicitly passed in --models are processed/plotted.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from section3_utils import MODEL_LABELS, ensure_dir, read_jsonl, safe_float as sf, safe_int as si

# -----------------------------
# Matplotlib "paper" styling (BIG fonts like your reference plot)
# -----------------------------
def set_paper_style():
    plt.rcParams.update(
        {
            # text
            "font.family": "DejaVu Sans",
            "font.size": 40,          
            "axes.titlesize": 40,      
            "axes.labelsize": 40,     
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": 35,
            "legend.title_fontsize": 40,

            # lines
            "lines.linewidth": 3.0,
            "lines.markersize": 8,

            # axes / figure
            "axes.linewidth": 1.2,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


X_ORDER = [-2, -1, 0, 1, 2]
X_TICKLABEL = {-2: "-2", -1: "-1", 0: "0", 1: "1", 2: "2"}


def shift_bin(class_diff: int) -> float:
    if class_diff <= -2:
        return -2
    if class_diff >= 2:
        return 2
    return float(class_diff)


# -----------------------------
# Load OR
# -----------------------------
def load_or_df(path: Path) -> pd.DataFrame:
    rows = []
    for o in read_jsonl(path):
        sid = si(o.get("subject_id"))
        hid = si(o.get("hadm_id"))
        ppl = sf(o.get("ppl"))
        if sid is None or hid is None or ppl is None:
            continue
        rows.append({"subject_id": sid, "hadm_id": hid, "ppl_or": ppl})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"OR file empty/invalid: {path}")
    print(f"[OR] loaded={len(df):,} from {path.name}")
    return df


# -----------------------------
# Load CF
# -----------------------------
def load_cf_df(path: Path) -> pd.DataFrame:
    rows = []
    skipped = 0

    for o in read_jsonl(path):
        sid = si(o.get("subject_id"))
        hid = si(o.get("hadm_id"))
        ppl = sf(o.get("ppl"))
        cd = si(o.get("class_diff"))
        #cd = si(o.get("class_diff_abs"))

        if sid is None or hid is None or ppl is None or cd is None:
            skipped += 1
            continue
        if cd < -4 or cd > 4:
            skipped += 1
            continue

        rows.append(
            {
                "subject_id": sid,
                "hadm_id": hid,
                "class_diff": cd,
                #"class_diff_abs": cd,
                "xbin": shift_bin(cd),
                "ppl_cf": ppl,
            }
        )

    df = pd.DataFrame(rows)
    print(f"[CF] loaded={len(df):,} from {path.name} | skipped={skipped:,}")
    return df


# -----------------------------
# Units + summary
# -----------------------------
def compute_units(df_or: pd.DataFrame, df_cf: pd.DataFrame) -> pd.DataFrame:
    df = df_cf.merge(df_or, on=["subject_id", "hadm_id"], how="inner")
    if df.empty:
        raise RuntimeError("CF/OR merge produced 0 rows. Check subject_id/hadm_id match.")

    df["dppl"] = df["ppl_cf"] - df["ppl_or"]

    # unit = per patient per binned shift (mean across CF samples)
    units = (
        df.groupby(["subject_id", "hadm_id", "xbin"], dropna=False)
        .agg(n_samples=("dppl", "size"), mean_dppl=("dppl", "mean"))
        .reset_index()
    )
    return units


def summarize_for_plot(units: pd.DataFrame) -> pd.DataFrame:
    g = units.groupby("xbin")["mean_dppl"]
    out = g.agg(["count", "mean", "std"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["ci_low"] = out["mean"] - 1.96 * out["se"]
    out["ci_high"] = out["mean"] + 1.96 * out["se"]
    return out


# -----------------------------
# Plot
# -----------------------------
def plot_lines(all_summary: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    set_paper_style()

    # BIG figure like your reference
    fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)

    for model, g in all_summary.groupby("model"):
        g = g.set_index("xbin").reindex(X_ORDER).reset_index()

        x = g["xbin"].to_numpy(dtype=float)
        y = g["mean"].to_numpy(dtype=float)
        lo = g["ci_low"].to_numpy(dtype=float)
        hi = g["ci_high"].to_numpy(dtype=float)

        ax.plot(x, y, marker="o", label=model)
        ax.fill_between(x, lo, hi, alpha=0.20)

    # reference lines
    #ax.axhline(0.0, linestyle="--", linewidth=2)

    # labels
    #ax.set_title("Average ΔPPL by severity shift (no-task)")
    ax.set_xlabel("Counterfactual severity shift")
    ax.set_ylabel("ΔPPL")

    # ticks
    ax.set_xticks(X_ORDER)
    ax.set_xticklabels([X_TICKLABEL[v] for v in X_ORDER])

    # Set y-axis limits to zoom in
    #ax.set_ylim(0, 0.17) 
    ax.set_ylim(0, 1.5) # Adjust this range based on ΔPPL data

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legend (bigger, clearer)
    #ax.legend(title=None, frameon=True, loc="best")
    ax.legend(
    title=None,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.28),
    ncol=2,
    frameon=False,
)

    # save
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, type=str)
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument(
        "--models",
        required=True,
        type=str,
        help="Comma-separated list of model keys exactly matching filenames (e.g. deepseek,gptoss120,llama,meditron,obllm,phi)",
    )
    ap.add_argument(
        "--or_template",
        default="{model}__task_independent__template__original__with_demographics.jsonl",
        type=str,
        help='Filename template for OR files (relative to results_dir). Use {model}.',
    )
    ap.add_argument(
        "--cf_template",
        default="{model}__task_independent__template__counterfactual__with_demographics.jsonl",
        type=str,
        help='Filename template for CF files (relative to results_dir). Use {model}.',
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"[models (STRICT)] {models}")

    summaries = []
    per_model_overall = []

    for m in models:
        or_path = results_dir / args.or_template.format(model=m)
        cf_path = results_dir / args.cf_template.format(model=m)

        if not or_path.exists():
            print(f"[WARN] missing OR file for model={m}: {or_path.name}")
            continue
        if not cf_path.exists():
            print(f"[WARN] missing CF file for model={m}: {cf_path.name}")
            continue

        print(f"\n=== {m} ===")
        df_or = load_or_df(or_path)
        df_cf = load_cf_df(cf_path)

        if df_cf.empty:
            print(f"[WARN] CF dataframe empty for model={m}")
            continue

        units = compute_units(df_or, df_cf)
        print(f"[units] n_units={len(units):,} (unique patient×shift bins)")

        summ = summarize_for_plot(units)
        #summ["model"] = m
        summ["model"] = MODEL_LABELS.get(m, m)
        summaries.append(summ)

        per_model_overall.append(
            {
                "model": MODEL_LABELS.get(m, m),
                "n_units": int(len(units)),
                "mean_dppl": float(np.nanmean(units["mean_dppl"].to_numpy(dtype=float))),
                "std_dppl": float(np.nanstd(units["mean_dppl"].to_numpy(dtype=float), ddof=1))
                if len(units) > 1
                else float("nan"),
            }
        )

    if not summaries:
        print("\nNo models produced usable results. Check filenames and --models keys.")
        return

    df_summary = pd.concat(summaries, ignore_index=True)
    df_summary.to_csv(out_dir / "ppl_shift_summary_by_shift.csv", index=False)

    df_overall = pd.DataFrame(per_model_overall).sort_values("model")
    df_overall.to_csv(out_dir / "ppl_shift_overall_by_model.csv", index=False)

    plot_lines(
        all_summary=df_summary,
        out_png=out_dir / "plot_avg_dppl_by_shift_notask_binned_extremes.png",
        out_pdf=out_dir / "plot_avg_dppl_by_shift_notask_binned_extremes.pdf",
    )

    print(f"\nDone. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
