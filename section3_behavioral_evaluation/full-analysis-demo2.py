#!/usr/bin/env python3
"""
Demographics CF analysis for LOS:

- OR-only task metrics (acc / macro-F1 / macro-precision / macro-recall) using labels from the Section 1 cohort file
- CF robustness vs OR with hierarchical aggregation (NO row-level outputs)
- KL divergence: KL(p_cf || p_or) as CF
- Expected LOS: E = p @ class_hours, where class_hours are PROVIDED via --override_class_hours
- Units: per (hadm×var×orig_group×cf_group): compute mean ΔE[LOS], flip%, mean KL

- One combined plot for ALL demographic categories (Age + Sex + Race) across models.
- Within each variable, for each category (e.g., Young adults), compare ΔE distribution vs "the rest"
  (other categories of the same variable), PER MODEL:
    - significance test on ΔE (Welch t-test by default)
    - optional FDR correction (BH) per model
  If NOT significant: leave the cell blank (like your example figure).
  If significant: show cell text "ΔE\n(%flip)".

ID FORMAT ASSUMPTION:
  <hadm_id>_<var>_<orig>_<cf>
Examples:
  21509857_sex_F_M
  21509857_race_white_black
  21509857_age_olderAdults_youngAdults
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from matplotlib.colors import LinearSegmentedColormap
from section3_utils import (
    MODEL_LABELS_HEATMAP as MODEL_LABELS,
    build_los_label_map_and_bucket_means,
    ensure_dir,
    read_jsonl,
    safe_float,
    safe_int,
)

# LaTeX RGB -> matplotlib expects 0..1 floats
reasoncolor = (85/255, 165/255, 185/255)
midcolor    = (0.96, 0.96, 0.96)
medcolor    = (225/255, 140/255, 75/255)


cmap = LinearSegmentedColormap.from_list(
    "reason_to_med",
    [reasoncolor, midcolor, medcolor],  # blue -> white -> orange
    N=256,
)

# -----------------------------
# Probs / KL
# -----------------------------
def probs_from_row_k4(row: dict) -> Optional[np.ndarray]:
    p = [safe_float(row.get(f"prob_{k}")) for k in (1, 2, 3, 4)]
    if any(v is None for v in p):
        return None
    arr = np.array(p, dtype=np.float64)
    s = arr.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    return arr / s


def kl_divergence(p_cf: np.ndarray, p_or: np.ndarray) -> float:
    p_cf = np.clip(p_cf, 1e-12, 1.0)
    p_or = np.clip(p_or, 1e-12, 1.0)
    p_cf = p_cf / p_cf.sum()
    p_or = p_or / p_or.sum()
    return float(np.sum(p_cf * np.log(p_cf / p_or)))


# -----------------------------
# Load OR
# -----------------------------
def load_or_df(or_path: Path) -> pd.DataFrame:
    rows = []
    for obj in read_jsonl(or_path):
        sid = obj.get("subject_id")
        hid = obj.get("hadm_id")
        if sid is None or hid is None:
            continue

        pred = safe_int(obj.get("pred_prob_class"))
        p = probs_from_row_k4(obj)
        if pred is None or p is None:
            continue

        rows.append(
            {
                "subject_id": int(sid),
                "hadm_id": int(hid),
                "pred_or": int(pred),
                "p1_or": float(p[0]),
                "p2_or": float(p[1]),
                "p3_or": float(p[2]),
                "p4_or": float(p[3]),
            }
        )

    df = pd.DataFrame(rows)
    print(f"[OR] loaded={len(df):,} from {or_path.name}")
    return df


# -----------------------------
# Task metrics (original-only)
# -----------------------------
def compute_task_metrics(df_or: pd.DataFrame, label_map: Dict[Tuple[int, int], int]) -> Dict[str, float]:
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
# Demographics categories + normalization
# -----------------------------
AGE_ORDER = ["youngAdults", "middleAgedAdults", "olderAdults", "elderly"]
AGE_LABEL = {
    "youngAdults": "Young\nadults",
    "middleAgedAdults": "Middle-aged\nadults",
    "olderAdults": "Older\nadults",
    "elderly": "Elderly",
}

SEX_LABEL = {"F": "Female", "M": "Male"}

RACE_CANON = ["Asian and Pacific", "Black", "Hispanic/Latino", "Other/Unknown", "White"]
RACE_LABEL = {
    "Asian and Pacific": "Asian and\nPacific",
    "Black": "Black",
    "Hispanic/Latino": "Hispanic/\nLatino",
    "Other/Unknown": "Other/\nUnknown",
    "White": "White",
}

RACE_MAP = {
    "white": "White",
    "caucasian": "White",
    "black": "Black",
    "africanamerican": "Black",
    "african_american": "Black",
    "asian": "Asian and Pacific",
    "asianandpacific": "Asian and Pacific",
    "asianandpacificislander": "Asian and Pacific",
    "asian_pacific": "Asian and Pacific",
    "asianpacific": "Asian and Pacific",
    "asianandpacificislanders": "Asian and Pacific",
    "hispanic": "Hispanic/Latino",
    "latino": "Hispanic/Latino",
    "hispaniclatino": "Hispanic/Latino",
    "hispanic/latino": "Hispanic/Latino",
    "other": "Other/Unknown",
    "unknown": "Other/Unknown",
    "otherunknown": "Other/Unknown",
    "other/unknown": "Other/Unknown",
}


def slugify(x: str) -> str:
    if x is None:
        return ""
    return (
        str(x)
        .strip()
        .replace(" ", "")
        .replace("-", "")
        .replace("/", "")
        .replace("_", "")
        .replace(".", "")
        .lower()
    )


def normalize_var_token(v: str) -> Optional[str]:
    if v is None:
        return None
    v = slugify(v)
    if v in ("sex", "gender"):
        return "sex"
    if v in ("race", "ethnicity", "ethnic"):
        return "race"
    if v == "age":
        return "age"
    return None


def normalize_age_group(x: str) -> Optional[str]:
    if x is None:
        return None
    s = slugify(x)
    if s in ("youngadults", "youngadult"):
        return "youngAdults"
    if s in ("middleagedadults", "middleagedadult", "middleageadults", "middleaged"):
        return "middleAgedAdults"
    if s in ("olderadults", "olderadult"):
        return "olderAdults"
    if s in ("elderly", "elder", "oldest"):
        return "elderly"
    return None


def normalize_sex(x: str) -> Optional[str]:
    if x is None:
        return None
    s = slugify(x)
    if s in ("f", "female", "woman"):
        return "F"
    if s in ("m", "male", "man"):
        return "M"
    return None


def normalize_race(x: str) -> str:
    s = slugify(x)
    return RACE_MAP.get(s, "Other/Unknown")


# -----------------------------
# Parse demographics id
# -----------------------------
def parse_demo_id(id_str: str) -> Optional[Tuple[int, str, str, str]]:
    if not id_str:
        return None
    parts = str(id_str).split("_")
    if len(parts) < 4:
        return None
    try:
        hadm = int(parts[0])
    except Exception:
        return None

    var = normalize_var_token(parts[1])
    if var is None:
        return None

    orig = parts[2]
    cf = parts[3]
    return hadm, var, orig, cf


def canonicalize_pair(var: str, orig_raw: str, cf_raw: str) -> Optional[Tuple[str, str, str]]:
    if var == "age":
        o = normalize_age_group(orig_raw)
        c = normalize_age_group(cf_raw)
        if o is None or c is None:
            return None
        return var, o, c

    if var == "sex":
        o = normalize_sex(orig_raw)
        c = normalize_sex(cf_raw)
        if o is None or c is None:
            return None
        return var, o, c

    if var == "race":
        o = normalize_race(orig_raw)
        c = normalize_race(cf_raw)
        return var, o, c

    return None


# -----------------------------
# Load CF demographics
# -----------------------------
def load_cf_demo_df(cf_path: Path) -> Tuple[pd.DataFrame, int, int]:
    rows = []
    skipped_unparsed_id = 0
    loaded_lines = 0

    for obj in read_jsonl(cf_path):
        loaded_lines += 1
        sid = obj.get("subject_id")
        hid = obj.get("hadm_id")
        if sid is None or hid is None:
            continue

        p = probs_from_row_k4(obj)
        pred = safe_int(obj.get("pred_prob_class"))
        if p is None or pred is None:
            continue

        parsed = parse_demo_id(obj.get("id"))
        if parsed is None:
            skipped_unparsed_id += 1
            continue
        _, var_raw, orig_raw, cf_raw = parsed

        canon = canonicalize_pair(var_raw, orig_raw, cf_raw)
        if canon is None:
            skipped_unparsed_id += 1
            continue
        var, orig_cat, cf_cat = canon

        rows.append(
            {
                "subject_id": int(sid),
                "hadm_id": int(hid),
                "var": var,
                "orig_cat": orig_cat,
                "cf_cat": cf_cat,
                "pred_cf": int(pred),
                "p1_cf": float(p[0]),
                "p2_cf": float(p[1]),
                "p3_cf": float(p[2]),
                "p4_cf": float(p[3]),
            }
        )

    df = pd.DataFrame(rows)
    print(f"[CF-demo] loaded={len(df):,} from {cf_path.name} | skipped_unparsed_id={skipped_unparsed_id:,}")
    return df, loaded_lines, skipped_unparsed_id


# -----------------------------
# Units: hadm×var×orig×cf
# -----------------------------
def compute_demo_units(df_or: pd.DataFrame, df_cf: pd.DataFrame, class_hours: np.ndarray) -> pd.DataFrame:
    df = df_cf.merge(df_or, on=["subject_id", "hadm_id"], how="inner")
    if df.empty:
        raise RuntimeError("CF/OR merge produced 0 rows. Check ID alignment.")

    p_or = df[["p1_or", "p2_or", "p3_or", "p4_or"]].to_numpy(dtype=np.float64)
    p_cf = df[["p1_cf", "p2_cf", "p3_cf", "p4_cf"]].to_numpy(dtype=np.float64)

    p_or = p_or / np.clip(p_or.sum(axis=1, keepdims=True), 1e-12, None)
    p_cf = p_cf / np.clip(p_cf.sum(axis=1, keepdims=True), 1e-12, None)

    df["kl"] = [kl_divergence(pc, po) for pc, po in zip(p_cf, p_or)]

    E_or = p_or @ class_hours
    E_cf = p_cf @ class_hours
    df["dE_hours"] = E_or - E_cf

    df["flip"] = (df["pred_cf"].astype(int) != df["pred_or"].astype(int)).astype(float)

    df_units = (
        df.groupby(["subject_id", "hadm_id", "var", "orig_cat", "cf_cat"], dropna=False)
        .agg(
            n=("kl", "size"),
            mean_kl=("kl", "mean"),
            mean_dE=("dE_hours", "mean"),
            pct_flip=("flip", lambda x: float(np.mean(x) * 100.0)),
        )
        .reset_index()
    )
    return df_units


# -----------------------------
# Cells for plotting: var×category×model (aggregate across cf targets)
# -----------------------------
def demo_cells_from_units(df_units_model: pd.DataFrame, model: str) -> pd.DataFrame:
    df = df_units_model.copy()
    df["model"] = model
    df_cells = (
        df.groupby(["var", "orig_cat"], dropna=False)
        .agg(
            mean_dE=("mean_dE", "mean"),
            pct_flip=("pct_flip", "mean"),
            n_units=("n", "sum"),
        )
        .reset_index()
    )
    df_cells["model"] = model
    df_cells = df_cells.rename(columns={"orig_cat": "category"})
    return df_cells


# -----------------------------
# Significance testing: category vs rest (within var), per model
# -----------------------------
def welch_t_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    try:
        from scipy.stats import ttest_ind
    except Exception as e:
        raise RuntimeError("scipy is required for significance testing. Install with: pip install scipy") from e
    return float(ttest_ind(a, b, equal_var=False).pvalue)


def fdr_bh(pvals: pd.Series) -> pd.Series:
    p = pvals.astype(float).to_numpy()
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return pd.Series(out, index=pvals.index)

    idx = np.argsort(p[finite])
    p_sorted = p[finite][idx]
    m = p_sorted.size
    q = p_sorted * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out_idx = np.where(finite)[0][idx]
    out[out_idx] = q
    return pd.Series(out, index=pvals.index)


def build_sig_table(
    df_units_all: pd.DataFrame,
    alpha: float = 0.05,
    fdr: bool = True,
    min_units_per_side: int = 10,
) -> Tuple[pd.DataFrame, str]:
    """
    df_units_all: hadm×var×orig×cf units + model, with mean_dE and pct_flip.
    We test on mean_dE for each (model,var,category=orig_cat) vs rest of that variable.
    """
    df = df_units_all.copy()
    df = df.rename(columns={"orig_cat": "category"})
    df = df[np.isfinite(df["mean_dE"])].copy()

    tests = []
    for (m, v), g in df.groupby(["model", "var"]):
        cats = g["category"].dropna().unique().tolist()
        for cat in cats:
            a = g.loc[g["category"] == cat, "mean_dE"].to_numpy(dtype=float)
            b = g.loc[g["category"] != cat, "mean_dE"].to_numpy(dtype=float)
            p = welch_t_pvalue(a, b)
            tests.append(
                {
                    "model": m,
                    "var": v,
                    "category": cat,
                    "p": p,
                    "n_cat": int(np.sum(np.isfinite(a))),
                    "n_rest": int(np.sum(np.isfinite(b))),
                }
            )

    test_df = pd.DataFrame(tests)
    if test_df.empty:
        test_df["q"] = np.nan
        return test_df, "p"

    if fdr:
        test_df["q"] = np.nan
        for m, gg in test_df.groupby("model"):
            test_df.loc[gg.index, "q"] = fdr_bh(gg["p"]).values
        sig_col = "q"
    else:
        sig_col = "p"

    test_df["is_sig"] = (
        (test_df[sig_col] < alpha) &
        (test_df["n_cat"] >= min_units_per_side) &
        (test_df["n_rest"] >= min_units_per_side)
    )

    return test_df, sig_col


# -----------------------------
# Combined plot: all variables together, blank if not significant
# -----------------------------
def row_order_and_label(var: str, cat: str) -> Tuple[int, str]:
    if var == "age":
        idx = AGE_ORDER.index(cat) if cat in AGE_ORDER else 999
        return 100 + idx, AGE_LABEL.get(cat, cat)
    if var == "sex":
        idx = 0 if cat == "F" else 1 if cat == "M" else 999
        return 200 + idx, SEX_LABEL.get(cat, cat)
    if var == "race":
        order = ["Asian and Pacific", "Black", "Hispanic/Latino", "Other/Unknown", "White"]
        idx = order.index(cat) if cat in order else 999
        return 300 + idx, RACE_LABEL.get(cat, cat)
    return 999, cat


def plot_combined_sig_heatmap(
    df_cells_all: pd.DataFrame,
    sig_df: pd.DataFrame,
    sig_col: str,
    models: List[str],
    out_png: Path,
    out_pdf: Optional[Path] = None,
    title: str = r'$\Delta \mathbb{E}[y_i^{\mathrm{los}}\mid\mathbf{x}_i]$ (%flip rate)',
) -> None:
    import seaborn as sns

    # -----------------------------
    # "paper" font scaling
    # -----------------------------
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 30,          # base
            "axes.titlesize": 30,
            "axes.labelsize": 30,
            "xtick.labelsize": 30,
            "ytick.labelsize": 30,
        }
    )

    cells = df_cells_all.copy()

    # merge significance flags
    sig_keep = sig_df[["model", "var", "category", "is_sig", sig_col]].copy()
    cells = cells.merge(sig_keep, on=["model", "var", "category"], how="left")
    cells["is_sig"] = cells["is_sig"].fillna(False)

    # row order + labels
    cells["row_order"] = cells.apply(lambda r: row_order_and_label(r["var"], r["category"])[0], axis=1)
    cells["row_label"] = cells.apply(lambda r: row_order_and_label(r["var"], r["category"])[1], axis=1)

    rows = (
        cells[["row_label", "row_order"]]
        .drop_duplicates()
        .sort_values("row_order")
    )
    row_labels_ordered = rows["row_label"].tolist()

    # pivots
    pivot_dE = (
        cells.pivot_table(index="row_label", columns="model", values="mean_dE", aggfunc="mean")
        .reindex(index=row_labels_ordered, columns=models)
    )
    pivot_flip = (
        cells.pivot_table(index="row_label", columns="model", values="pct_flip", aggfunc="mean")
        .reindex(index=row_labels_ordered, columns=models)
    )
    pivot_sig = (
        cells.pivot_table(index="row_label", columns="model", values="is_sig", aggfunc="max")
        .reindex(index=row_labels_ordered, columns=models)
    )

    # mask non-sig
    mask = ~pivot_sig.astype(bool)

    # annotations: only if significant
    ann = pd.DataFrame("", index=pivot_dE.index, columns=pivot_dE.columns)
    for r in ann.index:
        for c in ann.columns:
            if not bool(pivot_sig.loc[r, c]):
                continue
            de = pivot_dE.loc[r, c]
            fl = pivot_flip.loc[r, c]
            if np.isfinite(de) and np.isfinite(fl):
                ann.loc[r, c] = f"{de:.2f}\n({fl:.1f}%)"
            elif np.isfinite(de):
                ann.loc[r, c] = f"{de:.2f}\n(NA)"

    # -----------------------------
    # BIG figsize (explicit, not formula)
    # -----------------------------
    n_rows = pivot_dE.shape[0]
    n_cols = pivot_dE.shape[1]

    # tuned for your typical 11 rows (4 age + 2 sex + 5 race) and ~6-8 models
    fig_w = max(14, 1.8 * n_cols + 6)
    fig_h = max(10, 0.85 * n_rows + 5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    hm = sns.heatmap(
        pivot_dE,
        annot=ann,
        fmt="",
        cmap=cmap,
        center=0.0,
        linewidths=0.6,
        linecolor="white",
        mask=mask,
        cbar_kws={
            "label": r'$\Delta \mathbb{E}[y_i^{\mathrm{los}}\mid\mathbf{x}_i]$ (hours)',
            # right-side vertical bar is the default:
            # "orientation": "vertical",
            "pad": 0.02,     # small gap between heatmap and bar
            "shrink": 0.9,   # optional: slightly shorter than the heatmap
            "aspect": 25,    # optional: controls thickness (bigger = thinner)
        },
        annot_kws={"fontsize": 30},
        ax=ax,
    )



    # ticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, pad=14)

    # make colorbar text big too
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label(r'$\Delta \mathbb{E}[y_i^{\mathrm{los}}\mid\mathbf{x}_i]$ (hours)', fontsize=30)

    # separators between blocks: after 4 (age) and after 6 (age+sex)
    if n_rows >= 4:
        ax.hlines(4, *ax.get_xlim(), colors="black", linewidth=1.2)
    if n_rows >= 6:
        ax.hlines(6, *ax.get_xlim(), colors="black", linewidth=1.2)

    plt.tight_layout()

    # save
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight", transparent=True)

    plt.close(fig)



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
    ap.add_argument("--models", default="dsr1,llama3,meditron,obllm,phi4", type=str)

    ap.add_argument(
        "--or_template",
        default="{model}__los__template__original__with_demographics.jsonl",
        type=str,
        help='Filename template for OR files (relative to results_dir). Use {model}.',
    )
    ap.add_argument(
        "--cf_template",
        default="{model}__los__template__demographics_cf__with_demographics.jsonl",
        type=str,
        help='Filename template for demographics CF files (relative to results_dir). Use {model}.',
    )

    ap.add_argument(
        "--override_class_hours",
        default=None,
        type=str,
        help='Optional expected-LOS class hours as "h1,h2,h3,h4". If omitted, use cohort-derived mean hours per LOS bin.',
    )

    # significance params
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance threshold for p/q.")
    ap.add_argument("--no_fdr", action="store_true", help="Disable FDR correction (BH) per model.")
    ap.add_argument("--min_units_per_side", type=int, default=10, help="Min units in cat and rest to test/show cell.")

    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    labels_path = Path(args.labels_path)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    label_map, cohort_class_hours, _ = build_los_label_map_and_bucket_means(labels_path)

    if args.override_class_hours:
        class_hours = np.array(
            [float(x.strip()) for x in args.override_class_hours.split(",") if x.strip()],
            dtype=np.float64,
        )
        if class_hours.size != 4:
            raise ValueError("--override_class_hours must have exactly 4 values.")
        print(f"[expected LOS class_hours] using override: {class_hours.tolist()}")
    else:
        class_hours = cohort_class_hours
        print(f"[expected LOS class_hours] using cohort-derived means: {class_hours.tolist()}")

    all_cells = []
    all_units = []
    model_summary = []
    kl_boxplot_data = {}

    for m in models:
        model_label = MODEL_LABELS.get(m, m)
        or_path = results_dir / args.or_template.format(model=m)
        cf_path = results_dir / args.cf_template.format(model=m)

        if not or_path.exists():
            print(f"[WARN] missing OR file: {or_path}")
            continue
        if not cf_path.exists():
            print(f"[WARN] missing CF demo file: {cf_path}")
            continue

        print(f"\n=== {m} ===")
        df_or = load_or_df(or_path)
        task = compute_task_metrics(df_or, label_map)

        df_cf, _, _ = load_cf_demo_df(cf_path)
        if df_cf.empty:
            print(f"[WARN] CF demo dataframe empty for {m}.")
            continue

        df_units = compute_demo_units(df_or=df_or, df_cf=df_cf, class_hours=class_hours)
        #df_units["model"] = m
        df_units["model"] = model_label
        df_units["model_key"] = m  # recommended for traceability

        all_units.append(df_units)

        #df_cells = demo_cells_from_units(df_units, model=m)
        df_cells = demo_cells_from_units(df_units, model=model_label)

        all_cells.append(df_cells)

        #kl_boxplot_data[m] = df_units["mean_kl"].to_numpy(dtype=np.float64)
        kl_boxplot_data[model_label] = df_units["mean_kl"].to_numpy(dtype=np.float64)

        model_summary.append(
            {
                "model": model_label,
                "model_key": m,
                **task,
                "n_units": int(df_units.shape[0]),
                "avg_kl_units": float(df_units["mean_kl"].mean()),
                "std_kl_units": float(df_units["mean_kl"].std(ddof=1)) if df_units.shape[0] > 1 else float("nan"),
                "avg_dE_hours": float(df_units["mean_dE"].mean()),
                "std_dE_hours": float(df_units["mean_dE"].std(ddof=1)) if df_units.shape[0] > 1 else float("nan"),
                "pct_flip": float(df_units["pct_flip"].mean()),
            }
        )

        print(
            f"[TASK OR] n={task['n_or_labeled']:,} acc={task['accuracy']:.4f} f1={task['macro_f1']:.4f} "
            f"prec={task['macro_precision']:.4f} rec={task['macro_recall']:.4f}"
        )
        print(
            f"[DEMO units] n_units={df_units.shape[0]:,} avg_kl={df_units['mean_kl'].mean():.6f} "
            f"avg_ΔE={df_units['mean_dE'].mean():.3f}h avg_flip={df_units['pct_flip'].mean():.2f}%"
        )

    if not all_cells:
        print("\nNo models produced usable demographics results.")
        return

    df_cells_all = pd.concat(all_cells, ignore_index=True)
    df_units_all = pd.concat(all_units, ignore_index=True)

    df_cells_all.to_csv(out_dir / "demographics_cells.csv", index=False)
    df_units_all.to_csv(out_dir / "demographics_units.csv", index=False)

    summary_df = pd.DataFrame(model_summary).sort_values("model")
    summary_df.to_csv(out_dir / "summary_models_demographics.csv", index=False)

    # KL boxplot
    if kl_boxplot_data:
        try:
            import seaborn as sns
            rows = []
            for model, arr in kl_boxplot_data.items():
                arr = np.asarray(arr, dtype=float)
                arr = arr[np.isfinite(arr)]
                rows.extend([{"model": model, "kl": float(v)} for v in arr])
            df_box = pd.DataFrame(rows)
            if not df_box.empty:
                plt.figure(figsize=(10, 6))
                ax = sns.boxplot(data=df_box, x="model", y="kl", showfliers=False, color="#878A8F", width=0.6)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_ylabel("KL divergence  KL(p_cf || p_or)")
                ax.set_xlabel("")
                ax.set_title("Overall KL divergence per Model (demographics units)")
                plt.tight_layout()
                plt.savefig(out_dir / "plot_overall_kl_per_model_demographics.png", dpi=300)
                plt.close()
        except Exception as e:
            print(f"[WARN] Could not create KL boxplot: {e}")

    # Significance tests + combined plot
    sig_df, sig_col = build_sig_table(
        df_units_all=df_units_all,
        alpha=args.alpha,
        fdr=(not args.no_fdr),
        min_units_per_side=args.min_units_per_side,
    )
    sig_df.to_csv(out_dir / "significance_tests_demographics.csv", index=False)

    #models_present = sorted(df_cells_all["model"].unique().tolist())
    # preserve CLI order (using model labels)
    models_present = [MODEL_LABELS.get(m, m) for m in models]
    models_present = [m for m in models_present if m in df_cells_all["model"].unique()]


    plot_combined_sig_heatmap(
        df_cells_all=df_cells_all,
        sig_df=sig_df,
        sig_col=sig_col,
        models=models_present,
        out_png=out_dir / "plot_demo_ALLVARS_sig_deltaE_flip.png",
        out_pdf=out_dir / "plot_demo_ALLVARS_sig_deltaE_flip.pdf",
        title=r'$\Delta \mathbb{E}[y_i^{\mathrm{los}}\mid\mathbf{x}_i]$ (%flip rate)',
    )

    print(f"\nDone. Outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
