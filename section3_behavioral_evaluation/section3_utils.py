#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "llama": "LLaMA-3.3-Instruct-70B",
    "obllm": "OpenBioLLM-70B",
    "phi": "Phi4-14B",
    "meditron": "Meditron3-Phi4-70B",
    "deepseek": "DeepSeek-R1-Distill-70B",
    "gptoss120": "GPT-OSS-120B",
    "qwen25": "Qwen-2.5-Instruct-70B",
    "gpt41mini": "GPT-4.1-mini",
}

MODEL_LABELS_HEATMAP = {
    "llama": "LLaMA\n3.3-70B",
    "obllm": "OpenBioLLM\n70B",
    "phi": "Meditron3\nPhi4-14B",
    "meditron": "Meditron3\n70B",
    "deepseek": "DeepSeek\nR1-70B",
    "gptoss120": "GPT-OSS\n120B",
    "qwen25": "Qwen-2.5\n70B",
    "gpt41mini": "GPT-4.1\nmini",
}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def compute_los_quantiles(hours: Iterable[float]) -> tuple:
    arr = np.array(list(hours), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No usable LOS values available to compute quantiles.")
    q1, q2, q3 = np.quantile(arr, [0.25, 0.50, 0.75])
    return float(q1), float(q2), float(q3)


def los_hours_to_class(hours: float, q1: float, q2: float, q3: float) -> Optional[int]:
    if hours is None or not np.isfinite(hours):
        return None
    if hours < q1:
        return 1
    if hours < q2:
        return 2
    if hours < q3:
        return 3
    return 4


def load_labels_rows(labels_path: Path) -> List[dict]:
    if labels_path.suffix.lower() == ".csv":
        df = pd.read_csv(labels_path, low_memory=False)
        rename_map = {}
        if "episode_los_hours" in df.columns and "los_icu_hours" not in df.columns:
            rename_map["episode_los_hours"] = "los_icu_hours"
        if "mortality" in df.columns and "mortality_label" not in df.columns:
            rename_map["mortality"] = "mortality_label"
        if rename_map:
            df = df.rename(columns=rename_map)
        return df.to_dict(orient="records")
    return list(read_jsonl(labels_path))


def infer_labels_description(labels_path: Path) -> str:
    if labels_path.suffix.lower() == ".csv":
        return f"{labels_path.name} (Section 1 cohort CSV)"
    return labels_path.name


def build_mortality_label_map(labels_path: Path) -> Dict[tuple, int]:
    rows = load_labels_rows(labels_path)
    m: Dict[tuple, int] = {}
    missing = 0
    bad = 0
    dup = 0

    for obj in rows:
        sid = obj.get("subject_id")
        hid = obj.get("hadm_id")
        if sid is None or hid is None:
            continue

        y = obj.get("mortality_label")
        if y is None:
            missing += 1
            continue
        y = safe_int(y)
        if y is None or y not in (0, 1):
            bad += 1
            continue

        key = (int(sid), int(hid))
        if key in m:
            dup += 1
        m[key] = int(y)

    if not m:
        desc = infer_labels_description(labels_path)
        raise RuntimeError(f"No labels loaded. Check {desc} has subject_id/hadm_id/mortality_label.")

    print(
        f"[labels] source={infer_labels_description(labels_path)} | loaded={len(m):,} "
        f"| missing_mortality_label={missing:,} | bad_labels={bad:,} | duplicate_keys_overwritten={dup:,}"
    )
    return m


def build_los_label_map_and_bucket_means(labels_path: Path):
    rows = load_labels_rows(labels_path)
    label_map: Dict[tuple, int] = {}
    missing = 0
    unmapped = 0
    dup = 0
    bucket_vals = {1: [], 2: [], 3: [], 4: []}
    los_values = []

    for obj in rows:
        los_h = safe_float(obj.get("los_icu_hours"))
        if los_h is not None:
            los_values.append(float(los_h))

    q1, q2, q3 = compute_los_quantiles(los_values)

    for obj in rows:
        sid = obj.get("subject_id")
        hid = obj.get("hadm_id")
        if sid is None or hid is None:
            continue

        los_h = safe_float(obj.get("los_icu_hours"))
        if los_h is None:
            missing += 1
            continue

        y = los_hours_to_class(los_h, q1, q2, q3)
        if y is None:
            unmapped += 1
            continue

        key = (int(sid), int(hid))
        if key in label_map:
            dup += 1
        label_map[key] = int(y)
        bucket_vals[int(y)].append(float(los_h))

    if not label_map:
        desc = infer_labels_description(labels_path)
        raise RuntimeError(f"No labels loaded. Check {desc} has subject_id/hadm_id/los_icu_hours.")

    means = []
    for k in (1, 2, 3, 4):
        arr = np.array(bucket_vals[k], dtype=float)
        means.append(float(np.nanmean(arr)) if arr.size else float("nan"))
    class_hours_means = np.array(means, dtype=np.float64)

    if not np.all(np.isfinite(class_hours_means)):
        missing_classes = [str(idx) for idx, value in enumerate(class_hours_means, start=1) if not np.isfinite(value)]
        desc = infer_labels_description(labels_path)
        raise RuntimeError(
            f"Missing cohort-derived LOS class means for class(es) {', '.join(missing_classes)} in {desc}."
        )

    print(
        f"[labels] source={infer_labels_description(labels_path)} | loaded={len(label_map):,} "
        f"| missing_los_icu_hours={missing:,} | unmapped={unmapped:,} | duplicate_keys_overwritten={dup:,}"
    )
    print(f"[los quantiles hours] q1={q1:.6f}, q2={q2:.6f}, q3={q3:.6f}")
    print(
        "[bucket means hours] "
        + ", ".join([f"class{k}={class_hours_means[k-1]:.6f}" for k in (1, 2, 3, 4)])
    )
    return label_map, class_hours_means, (q1, q2, q3)
