#!/usr/bin/env python3
"""
LOS: OR-only task metrics + CF robustness vs OR with hierarchical aggregation
(using LOS quartile buckets in HOURS) + ΔE[LOS] (HOURS) and KL(p_cf || p_or)
by signed severity plot with 95% CI.

- Plots severity shifts in [-2, -1, 0, 1, 2]
- Uses big fonts + big figure size
- Saves both png and pdf for KL and ΔE plots
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from section3_utils import (
    MODEL_LABELS,
    build_los_label_map_and_bucket_means,
    ensure_dir,
    read_jsonl,
    safe_float,
    safe_int,
)

# -----------------------------
# Plot range: ONLY show -2..2
# -----------------------------
SEV_MIN = -2
SEV_MAX = 2
SEV_TICKS = [-2, -1, 0, 1, 2]


# -----------------------------
# Matplotlib "paper" styling (BIG like your reference plot)
# -----------------------------
def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 40,
            "axes.titlesize": 40,
            "axes.labelsize": 40,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
            "legend.fontsize": 35,
            "legend.title_fontsize": 40,
            "lines.linewidth": 3.0,
            "lines.markersize": 8,
            "axes.linewidth": 1.2,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )

# -----------------------------
# Probs / KL / sign
# -----------------------------
def probs_from_row(row: dict) -> Optional[np.ndarray]:
    p = [safe_float(row.get(f"prob_{k}")) for k in (1, 2, 3, 4)]
    if any(v is None for v in p):
        return None
    arr = np.array(p, dtype=np.float64)
    s = arr.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    return arr / s


def kl_divergence(p_cf: np.ndarray, p_or: np.ndarray) -> float:
    """KL(p_cf || p_or) = sum p_cf * log(p_cf / p_or)."""
    p_cf = np.clip(p_cf, 1e-12, 1.0)
    p_or = np.clip(p_or, 1e-12, 1.0)
    p_cf = p_cf / p_cf.sum()
    p_or = p_or / p_or.sum()
    return float(np.sum(p_cf * np.log(p_cf / p_or)))


def sgn(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


# -----------------------------
# Load OR (originals) per model
# -----------------------------
def load_or_df(or_path: Path) -> pd.DataFrame:
    rows = []
    for obj in read_jsonl(or_path):
        sid = obj.get("subject_id")
        hid = obj.get("hadm_id")
        if sid is None or hid is None:
            continue

        pred = safe_int(obj.get("pred_prob_class"))
        p = probs_from_row(obj)

        rows.append(
            {
                "subject_id": int(sid),
                "hadm_id": int(hid),
                "pred_or": pred,
                "p1_or": p[0] if p is not None else np.nan,
                "p2_or": p[1] if p is not None else np.nan,
                "p3_or": p[2] if p is not None else np.nan,
                "p4_or": p[3] if p is not None else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["pred_or", "p1_or", "p2_or", "p3_or", "p4_or"]).copy()
    df["pred_or"] = df["pred_or"].astype(int)

    print(f"[OR] loaded={len(df):,} from {or_path.name}")
    return df


# -----------------------------
# Task metrics computed only on OR
# -----------------------------
def compute_task_metrics(df_or: pd.DataFrame, label_map: Dict[tuple[int, int], int]) -> Dict[str, float]:
    y_true = []
    y_pred = []

    for r in df_or.itertuples(index=False):
        key = (int(r.subject_id), int(r.hadm_id))
        if key in label_map:
            y_true.append(int(label_map[key]))
            y_pred.append(int(r.pred_or))

    if not y_true:
        return {
            "n_or_labeled": 0,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
        }

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    return {
        "n_or_labeled": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# -----------------------------
# Load CF per model
# -----------------------------
def load_cf_df(cf_path: Path) -> pd.DataFrame:
    rows = []
    for obj in read_jsonl(cf_path):
        sid = obj.get("subject_id")
        hid = obj.get("hadm_id")
        if sid is None or hid is None:
            continue

        p = probs_from_row(obj)
        cda = safe_int(obj.get("class_diff_abs"))  # SIGNED (per your definition)

        rows.append(
            {
                "subject_id": int(sid),
                "hadm_id": int(hid),
                "vital": obj.get("vital"),
                "class_diff_abs": cda,  # signed severity delta
                "pred_cf": safe_int(obj.get("pred_prob_class")),
                "p1_cf": p[0] if p is not None else np.nan,
                "p2_cf": p[1] if p is not None else np.nan,
                "p3_cf": p[2] if p is not None else np.nan,
                "p4_cf": p[3] if p is not None else np.nan,
                "id": obj.get("id"),
            }
        )

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["p1_cf", "p2_cf", "p3_cf", "p4_cf"]).copy()
    print(f"[CF] loaded={len(df):,} from {cf_path.name}")
    return df


# -----------------------------
# Robustness helpers
# -----------------------------
def mono_pct_sign_aware(sev: np.ndarray, mean_dE: np.ndarray) -> float:
    """
    Sign-aware monotonicity for mean ΔE vs signed severity.

    Checks:
      A) Sign constraint (excluding sev==0):
         sev > 0 => mean_dE > 0
         sev < 0 => mean_dE < 0
      B) Monotonicity across adjacent severities:
         mean_dE non-decreasing as sev increases.
    """
    mask = np.isfinite(sev) & np.isfinite(mean_dE)
    sev = sev[mask].astype(float)
    mean_dE = mean_dE[mask].astype(float)

    if sev.size == 0:
        return float("nan")

    idx = np.argsort(sev)
    sev = sev[idx]
    mean_dE = mean_dE[idx]

    checks = []
    for s, de in zip(sev, mean_dE):
        if s == 0:
            continue
        checks.append(de > 0 if s > 0 else de < 0)

    if mean_dE.size >= 2:
        checks.extend(list(np.diff(mean_dE) >= -1e-12))

    if len(checks) == 0:
        return float("nan")
    return float(100.0 * np.mean(checks))


def robustness_analysis(
    df_or: pd.DataFrame,
    df_cf: pd.DataFrame,
    class_hours: np.ndarray,
):
    """
    Returns:
      df_indiv_vital: per (subject_id, hadm_id, vital) aggregation
      df_indiv_overall: per (subject_id, hadm_id) aggregation
      summary: dict with robustness summaries (hadm-level)
      kl_for_boxplot: per (hadm×vital) mean KL values
      df_pvs: per (hadm×vital×severity) mean dE_hours and mean kl (for CI plots)
    """
    df = df_cf.merge(df_or, on=["subject_id", "hadm_id"], how="inner")
    if df.empty:
        raise RuntimeError("CF/OR merge produced 0 rows. Check ID alignment.")

    # normalize probs
    p_or = df[["p1_or", "p2_or", "p3_or", "p4_or"]].to_numpy(dtype=np.float64)
    p_cf = df[["p1_cf", "p2_cf", "p3_cf", "p4_cf"]].to_numpy(dtype=np.float64)

    p_or = p_or / np.clip(p_or.sum(axis=1, keepdims=True), 1e-12, None)
    p_cf = p_cf / np.clip(p_cf.sum(axis=1, keepdims=True), 1e-12, None)

    # row-level KL (only for aggregation)
    df["kl"] = [kl_divergence(pc, po) for pc, po in zip(p_cf, p_or)]

    # expected LOS shift
    E_or = p_or @ class_hours
    E_cf = p_cf @ class_hours
    df["dE_hours"] = E_cf - E_or

    df["flip"] = np.where(
        df["pred_or"].notna() & df["pred_cf"].notna(),
        (df["pred_or"].astype(float) != df["pred_cf"].astype(float)).astype(float),
        np.nan,
    )

    def corr_dir(row) -> float:
        sev = row["class_diff_abs"]
        if sev is None or pd.isna(sev):
            return np.nan
        sd = sgn(float(row["dE_hours"]))
        ss = sgn(float(sev))
        if sd == 0 or ss == 0:
            return np.nan
        return 1.0 if sd == ss else 0.0

    df["correct_dir"] = df.apply(corr_dir, axis=1)

    # Step 1: Aggregate within (subject_id, hadm_id, vital)
    df_indiv_vital = (
        df.groupby(["subject_id", "hadm_id", "vital"], dropna=False)
        .agg(
            n_cf=("kl", "size"),
            mean_kl=("kl", "mean"),
            mean_flip=("flip", "mean"),
            mean_correct_dir=("correct_dir", "mean"),
            mean_dE=("dE_hours", "mean"),
        )
        .reset_index()
    )

    # Step 2: Aggregate within (subject_id, hadm_id) by averaging over vitals
    df_indiv_overall = (
        df_indiv_vital.groupby(["subject_id", "hadm_id"], dropna=False)
        .agg(
            mean_kl=("mean_kl", "mean"),
            mean_flip=("mean_flip", "mean"),
            mean_correct_dir=("mean_correct_dir", "mean"),
            mean_dE=("mean_dE", "mean"),
        )
        .reset_index()
    )

    # Per (hadm×vital×sev): mean KL + mean dE (this is what we plot with CI)
    df_pvs = (
        df.dropna(subset=["class_diff_abs"])
        .groupby(["subject_id", "hadm_id", "vital", "class_diff_abs"], dropna=False)
        .agg(
            kl=("kl", "mean"),
            dE_hours=("dE_hours", "mean"),
        )
        .reset_index()
    )

    # Hadm-level summaries
    pct_mono = mono_pct_sign_aware(
        sev=df_pvs.groupby("class_diff_abs")["dE_hours"].mean().index.to_numpy(dtype=np.float64),
        mean_dE=df_pvs.groupby("class_diff_abs")["dE_hours"].mean().to_numpy(dtype=np.float64),
    )

    summary = {
        "n_patients": int(df_indiv_overall.shape[0]),
        "n_indiv_vital_groups": int(df_indiv_vital.shape[0]),
        "avg_kl_hadm": float(df_indiv_overall["mean_kl"].mean()),
        "std_kl_hadm": float(df_indiv_overall["mean_kl"].std(ddof=1)) if df_indiv_overall.shape[0] > 1 else float("nan"),
        "pct_flip": float(df_indiv_overall["mean_flip"].mean() * 100.0),
        "pct_correct_dir": float(df_indiv_overall["mean_correct_dir"].mean() * 100.0),
        "pct_mono": float(pct_mono),
        "avg_dE_hours": float(df_indiv_overall["mean_dE"].mean()),
        "std_dE_hours": float(df_indiv_overall["mean_dE"].std(ddof=1)) if df_indiv_overall.shape[0] > 1 else float("nan"),
    }

    # Boxplot units: (hadm×vital) mean KL
    kl_for_boxplot = df_indiv_vital["mean_kl"].to_numpy(dtype=np.float64)

    return df_indiv_vital, df_indiv_overall, summary, kl_for_boxplot, df_pvs


# -----------------------------
# Plots
# -----------------------------
def plot_metric_by_severity_ci_big(
    df_all_pvs: pd.DataFrame,
    metric_col: str,
    out_png: Path,
    out_pdf: Path,
    out_csv: Path,
    title: str,
    ylabel: str,
) -> None:
    df_all_pvs = df_all_pvs.dropna(subset=["class_diff_abs"]).copy()
    df_all_pvs = df_all_pvs[df_all_pvs["class_diff_abs"].between(SEV_MIN, SEV_MAX)]

    agg = (
        df_all_pvs.groupby(["model", "class_diff_abs"])[metric_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"class_diff_abs": "sev"})
    )

    agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    agg["ci95"] = 1.96 * agg["se"]
    agg.to_csv(out_csv, index=False)

    set_paper_style()
    fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)

    # stable order
    models = list(agg["model"].unique())

    for model in models:
        g = agg[agg["model"] == model].sort_values("sev")
        x = g["sev"].to_numpy(dtype=float)
        y = g["mean"].to_numpy(dtype=float)
        ci = g["ci95"].to_numpy(dtype=float)

        ax.plot(x, y, marker="o", label=model)
        ax.fill_between(x, y - ci, y + ci, alpha=0.20, zorder=-1)

    ax.axhline(0, ls="--", lw=2, c="grey")
    ax.axvline(0, ls="--", lw=2, c="grey")

    ax.set_title(title)
    ax.set_xlabel("Counterfactual severity shift")
    ax.set_ylabel(ylabel)

    ax.set_xlim(SEV_MIN, SEV_MAX)
    ax.set_xticks(SEV_TICKS)

    # Set y-axis limits for both plots
    if metric_col == "kl":
        ax.set_ylim(0.0, 0.15)
    elif metric_col == "dE_hours":
        ax.set_ylim(-5, 5)  # Adjust this range based on your data

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # NO legend title (matches what you wanted)
    #ax.legend(title=None, frameon=True, loc="best", ncol=2)
    ax.legend(
    title=None,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.28),
    ncol=2,
    frameon=False,
)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_overall_kl_boxplot_matplotlib(kl_by_model: Dict[str, np.ndarray], out_png: Path, out_pdf: Path) -> None:
    """Matplotlib-only boxplot (per hadm×vital mean KL per model)."""
    rows = []
    for model, arr in kl_by_model.items():
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        for v in arr:
            rows.append((model, float(v)))

    if not rows:
        print("[WARN] No KL data to plot boxplot.")
        return

    df = pd.DataFrame(rows, columns=["model", "kl"])
    models = sorted(df["model"].unique().tolist())

    #data = [df.loc[df["model"] == m, "kl"].to_numpy(dtype=float) for m in models]
    data = [df[df["model"] == m]["kl"].to_numpy(dtype=float) for m in models]

    set_paper_style()
    fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)
    ax.boxplot(data, labels=models, showfliers=False)

    ax.set_ylabel("KL")
    ax.set_title("Overall KL divergence per model (hadm×vital units)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)


# -----------------------------
# Model family overview
# -----------------------------
MODEL_FAMILY = {
    "obllm": "medical",
    "meditron": "medical",
    "llama": "general_purpose",
    "phi": "general_purpose",
    "deepseek": "thinking",
    "qwen25": "general_purpose",
    "gptoss120": "thinking",
}


def category_overview(summary_df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Mean/std across models per family for selected metrics."""
    if summary_df.empty:
        return summary_df

    df = summary_df.copy()
    df["family"] = df["model_key"].map(MODEL_FAMILY).fillna("other")

    metrics = [
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "avg_kl_hadm",
        "pct_correct_dir",
        "pct_flip",
        "pct_mono",
        "avg_dE_hours",
    ]
    metrics = [m for m in metrics if m in df.columns]

    out_rows = []
    for fam, g in df.groupby("family"):
        row = {"family": fam, "n_models": int(g.shape[0])}
        for m in metrics:
            row[f"{m}_mean"] = float(np.nanmean(g[m].to_numpy(dtype=float)))
            row[f"{m}_std"] = float(np.nanstd(g[m].to_numpy(dtype=float), ddof=1)) if g.shape[0] > 1 else float("nan")
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows).sort_values(["family"])
    out_df.to_csv(out_csv, index=False)
    return out_df


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, type=str)
    ap.add_argument(
        "--labels_path",
        required=True,
        type=str,
        help="Path to Section 1 cohort labels: icu_cohort_data.csv, or a legacy JSONL with subject_id/hadm_id/los_icu_hours.",
    )
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--models", default="obllm,llama,phi,meditron,deepseek,gptoss120,qwen25", type=str)
    ap.add_argument(
        "--or_template",
        default="{model}__los__raw__original__no_demographics.jsonl",
        type=str,
        help='Filename template for OR files (relative to results_dir). Use {model}.',
    )
    ap.add_argument(
        "--cf_template",
        default="{model}__los__raw__counterfactual__no_demographics.jsonl",
        type=str,
        help='Filename template for CF files (relative to results_dir). Use {model}.',
    )

    ap.add_argument(
        "--override_class_hours",
        default="",
        type=str,
        help='Override expected-LOS class hours as "h1,h2,h3,h4". If set, ignores bucket means.',
    )

    ap.add_argument("--save_indiv_vital", action="store_true")
    ap.add_argument("--save_indiv_hadm", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    labels_path = Path(args.labels_path)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Labels + bucket means
    label_map, bucket_means, _ = build_los_label_map_and_bucket_means(labels_path)

    # Expected LOS class representative hours
    if args.override_class_hours.strip():
        class_hours = np.array(
            [float(x.strip()) for x in args.override_class_hours.split(",") if x.strip()],
            dtype=np.float64,
        )
        if class_hours.size != 4:
            raise ValueError("--override_class_hours must have exactly 4 values.")
        print(f"[expected LOS class_hours OVERRIDE] {class_hours.tolist()}")
    else:
        class_hours = bucket_means.astype(np.float64)
        print(f"[expected LOS class_hours (cohort-derived bucket means)] {class_hours.tolist()}")

    model_rows = []
    kl_boxplot_data: Dict[str, np.ndarray] = {}
    all_pvs_rows = []

    for m in models:
        or_path = results_dir / args.or_template.format(model=m)
        cf_path = results_dir / args.cf_template.format(model=m)

        if not or_path.exists():
            print(f"[WARN] missing OR file: {or_path}")
            continue
        if not cf_path.exists():
            print(f"[WARN] missing CF file: {cf_path}")
            continue

        print(f"\n=== {m} ===")
        df_or = load_or_df(or_path)
        task = compute_task_metrics(df_or, label_map)

        df_cf = load_cf_df(cf_path)
        df_indiv_vital, df_indiv_overall, robust, kl_vals, df_pvs = robustness_analysis(
            df_or=df_or,
            df_cf=df_cf,
            class_hours=class_hours,
        )

        # collect for plots
        kl_boxplot_data[MODEL_LABELS.get(m, m)] = kl_vals
        tmp = df_pvs.copy()
        #tmp["model"] = m
        tmp["model"] = MODEL_LABELS.get(m, m)
        all_pvs_rows.append(tmp)

        # model summary row
        model_rows.append(
    {"model_key": m, "model": MODEL_LABELS.get(m, m), **task, **robust}
)

        # optional saves
        if args.save_indiv_vital:
            df_indiv_vital.to_csv(out_dir / f"per_individual_vital_{m}.csv", index=False)
        if args.save_indiv_hadm:
            df_indiv_overall.to_csv(out_dir / f"per_hadm_overall_{m}.csv", index=False)

        print(
            f"[TASK OR] n={task['n_or_labeled']:,} acc={task['accuracy']:.4f} f1={task['macro_f1']:.4f} "
            f"prec={task['macro_precision']:.4f} rec={task['macro_recall']:.4f}"
        )
        print(
            f"[ROBUST hadm-level] patients={robust['n_patients']:,} "
            f"avg_kl_hadm={robust['avg_kl_hadm']:.6f} std_kl_hadm={robust['std_kl_hadm']:.6f} "
            f"flip={robust['pct_flip']:.2f}% corrdir={robust['pct_correct_dir']:.2f}% mono(ΔE)={robust['pct_mono']:.2f}% "
            f"avg_ΔE={robust['avg_dE_hours']:.3f}h std_ΔE={robust['std_dE_hours']:.3f}h"
        )

    # -----------------------------
    # Write summary CSVs
    # -----------------------------
    summary_df = pd.DataFrame(model_rows)
    if not summary_df.empty:
        cols = [
            "model_key",
            "model",
            "accuracy",
            "macro_f1",
            "macro_precision",
            "macro_recall",
            "avg_kl_hadm",
            "std_kl_hadm",
            "pct_correct_dir",
            "pct_flip",
            "pct_mono",
            "avg_dE_hours",
            "std_dE_hours",
            "n_or_labeled",
            "n_patients",
            "n_indiv_vital_groups",
        ]
        cols = [c for c in cols if c in summary_df.columns]
        summary_df = summary_df[cols].sort_values("model")
        summary_df.to_csv(out_dir / "summary_models.csv", index=False)

        cat_df = category_overview(summary_df, out_dir / "category_overview.csv")
        print("\n[category overview]")
        print(cat_df.to_string(index=False))

    # -----------------------------
    # Plots + CI CSVs
    # -----------------------------
    if kl_boxplot_data:
        plot_overall_kl_boxplot_matplotlib(
            kl_by_model=kl_boxplot_data,
            out_png=out_dir / "plot_overall_kl_per_model.png",
            out_pdf=out_dir / "plot_overall_kl_per_model.pdf",
        )

    if all_pvs_rows:
        df_all_pvs = pd.concat(all_pvs_rows, ignore_index=True)

        # save per-unit data (hadm×vital×sev)
        df_all_pvs.to_csv(out_dir / "patient_vital_severity_all_models.csv", index=False)

        # ΔE plot + CI csv (PNG + PDF)
        plot_metric_by_severity_ci_big(
            df_all_pvs=df_all_pvs,
            metric_col="dE_hours",
            out_png=out_dir / "plot_dE_hours_severity_per_model.png",
            out_pdf=out_dir / "plot_dE_hours_severity_per_model.pdf",
            out_csv=out_dir / "dE_hours_by_severity_per_model.csv",
            #title="Average ΔE[LOS] by severity per model",
            title=None,
            #ylabel="ΔE[LOS] (hours)",
            ylabel=r'$\Delta \mathbb{E}[y_i^{\mathrm{los}}\mid\mathbf{x}_i]$',
        )

        # KL plot + CI csv (PNG + PDF)
        plot_metric_by_severity_ci_big(
            df_all_pvs=df_all_pvs,
            metric_col="kl",
            out_png=out_dir / "plot_kl_severity_per_model.png",
            out_pdf=out_dir / "plot_kl_severity_per_model.pdf",
            out_csv=out_dir / "kl_by_severity_per_model.csv",
            #title="Average KL by severity per model",
            title=None,
            ylabel="KL",
        )

    print(f"\nDone. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
