#!/usr/bin/env python3

#input: cohort CSV, cleaned vitals JSONL, per-vital counterfactual JSONLs
#output: original and counterfactual template-based notes JSONL (vital-based and demographic-based)

import argparse
from pathlib import Path
import random
from typing import Any, Dict, List

from section1_utils import (
    DEFAULT_DATA_DIR,
    DEMOGRAPHIC_AGE_CATEGORIES,
    DEMOGRAPHIC_RACE_CLASSES,
    VITAL_KEYS,
    VITAL_TO_CODE,
    demographic_age_category,
    get_class_severity,
    load_jsonl,
    normalize_class_name,
    sample_demographic_ages,
    write_jsonl,
)


# ============================================================
def normalize_sex(val: Any) -> str:
    if val is None:
        return "NaN"
    s = str(val).strip()
    if s == "":
        return "NaN"
    return s


def normalize_race(val: Any) -> str:
    if val is None:
        return "other/unknown"
    s = str(val).strip()
    if s == "":
        return "other/unknown"
    return s.lower()


def normalize_age(val: Any) -> Any:
    if val is None:
        return "NaN"
    s = str(val).strip()
    if s == "":
        return "NaN"
    try:
        f = float(val)
        if abs(f - round(f)) < 1e-9:
            return int(round(f))
        return f
    except Exception:
        return val


def default_vitals() -> Dict[str, str]:
    return {
        "temperature": "NaN",
        "heart_rate": "NaN",
        "blood_pressure": "NaN",
        "respiration_rate": "NaN",
        "oxygen_saturation": "NaN",
    }


def clone_template_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "subject_id": row["subject_id"],
        "hadm_id": row["hadm_id"],
        "age": row["age"],
        "sex": row["sex"],
        "race": row["race"],
        "vitals": dict(row["vitals"]),
        "los_icu_hours": row["los_icu_hours"],
        "mortality_label": row["mortality_label"],
    }


# ============================================================
# LOAD COHORT / DEMOGRAPHICS
# ============================================================

def load_demo_data(cohort_csv: Path) -> Dict[tuple, Dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(cohort_csv, low_memory=False)

    demo = {}
    for _, row in df.iterrows():
        key = (int(row["subject_id"]), int(row["hadm_id"]))
        demo[key] = {
            "age": normalize_age(row.get("age")),
            "sex": normalize_sex(row.get("gender")),
            "race": normalize_race(row.get("race_group")),
            "los_icu_hours": float(row["episode_los_hours"]) if row.get("episode_los_hours") is not None else None,
            "mortality_label": int(row["mortality"]) if row.get("mortality") is not None else None,
        }
    return demo


# ============================================================
# LOAD CLEANED VITALS
# ============================================================

def load_cleaned_vitals(path: Path) -> Dict[tuple, Dict[str, str]]:
    rows = load_jsonl(path)
    lookup = {}

    for row in rows:
        key = (int(row["subject_id"]), int(row["hadm_id"]))
        vitals = row.get("vitals", {}) or {}

        merged = default_vitals()
        for vital in VITAL_KEYS:
            merged[vital] = vitals.get(vital, "NaN")

        lookup[key] = merged

    return lookup


# ============================================================
# BUILD ORIGINAL TEMPLATE NOTES
# ============================================================

def build_original_templates(
    demo_data: Dict[tuple, Dict[str, Any]],
    vitals_data: Dict[tuple, Dict[str, str]],
) -> List[Dict[str, Any]]:
    rows = []

    all_keys = sorted(set(demo_data.keys()) & set(vitals_data.keys()))

    for key in all_keys:
        subject_id, hadm_id = key
        demo = demo_data.get(key, {})
        race_class = demo.get("race", "other/unknown")

        template = {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "age": demo.get("age", "NaN"),
            "sex": demo.get("sex", "NaN"),
            "race": race_class,
            "vitals": vitals_data.get(key, default_vitals()),
            "los_icu_hours": demo.get("los_icu_hours"),
            "mortality_label": demo.get("mortality_label"),
        }
        rows.append(template)

    return rows


# ============================================================
# LOAD COUNTERFACTUAL DICTIONARIES
# ============================================================

def load_cf_lookup(cf_path: Path) -> Dict[int, Dict[str, Any]]:
    rows = load_jsonl(cf_path)
    return {int(row["hadm_id"]): row for row in rows}


# ============================================================
# BUILD COUNTERFACTUAL TEMPLATE NOTES
# ============================================================

def build_counterfactual_templates(
    original_templates: List[Dict[str, Any]],
    cf_paths: Dict[str, Path],
) -> List[Dict[str, Any]]:
    originals_by_hadm = {int(row["hadm_id"]): row for row in original_templates}
    all_cf_rows = []

    for vital_type, cf_path in cf_paths.items():
        cf_lookup = load_cf_lookup(cf_path)

        for hadm_id, cf_entry in cf_lookup.items():
            original_template = originals_by_hadm.get(hadm_id)
            if original_template is None:
                continue

            original_class = cf_entry.get("original", {}).get("original_class")
            original_severity = get_class_severity(vital_type, original_class)

            for aug in cf_entry.get("augmentations", []):
                counterfactual_class = aug.get("new_class")
                counterfactual_severity = get_class_severity(vital_type, counterfactual_class)

                if original_severity is not None and counterfactual_severity is not None:
                    class_diff = counterfactual_severity - original_severity
                    class_diff_abs = abs(counterfactual_severity) - abs(original_severity)
                else:
                    class_diff = None
                    class_diff_abs = None

                for aug_key, new_value in aug.items():
                    if not aug_key.startswith("new_value_"):
                        continue

                    cf_template = clone_template_row(original_template)

                    cf_template["vitals"][vital_type] = new_value

                    safe_class = str(counterfactual_class).lower().replace(" ", "_").replace("/", "_")
                    cf_template["id"] = (
                        f"{hadm_id}_{VITAL_TO_CODE[vital_type]}_{safe_class}_{aug_key.split('_')[-1]}"
                    )
                    cf_template["original_class"] = normalize_class_name(original_class)
                    cf_template["original_severity"] = original_severity
                    cf_template["counterfactual_class"] = normalize_class_name(counterfactual_class)
                    cf_template["counterfactual_severity"] = counterfactual_severity
                    cf_template["class_diff"] = class_diff
                    cf_template["class_diff_abs"] = class_diff_abs

                    all_cf_rows.append(cf_template)

    return all_cf_rows


def build_demographic_counterfactual_templates(
    original_templates: List[Dict[str, Any]],
    seed: int = 42,
    n_same_age_samples: int = 5,
    n_cross_age_samples: int = 5,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    counterfactuals: List[Dict[str, Any]] = []

    for template in original_templates:
        hadm_id = template["hadm_id"]

        orig_sex = template.get("sex", "NaN")
        if orig_sex in {"F", "M"}:
            new_sex = "M" if orig_sex == "F" else "F"
            cf = clone_template_row(template)
            cf["sex"] = new_sex
            cf["id"] = f"{hadm_id}_sex_{orig_sex}_{new_sex}"
            counterfactuals.append(cf)

        orig_race = str(template.get("race", "other/unknown")).strip().lower() or "other/unknown"
        for new_race in DEMOGRAPHIC_RACE_CLASSES:
            if new_race == orig_race:
                continue
            cf = clone_template_row(template)
            cf["race"] = new_race
            cf["id"] = f"{hadm_id}_race_{orig_race}_{new_race}"
            counterfactuals.append(cf)

        orig_age = template.get("age")
        orig_category = demographic_age_category(orig_age)
        if orig_category is None:
            continue

        for new_age in sample_demographic_ages(orig_category, rng=rng, original_age=orig_age, n=n_same_age_samples):
            cf = clone_template_row(template)
            cf["age"] = new_age
            cf["id"] = f"{hadm_id}_age_{orig_category}_{orig_category}"
            counterfactuals.append(cf)

        for new_category in DEMOGRAPHIC_AGE_CATEGORIES.keys():
            if new_category == orig_category:
                continue
            for new_age in sample_demographic_ages(new_category, rng=rng, original_age=None, n=n_cross_age_samples):
                cf = clone_template_row(template)
                cf["age"] = new_age
                cf["id"] = f"{hadm_id}_age_{orig_category}_{new_category}"
                counterfactuals.append(cf)

    return counterfactuals


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Build template-based original and counterfactual notes")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Base directory for Section 1 data")
    parser.add_argument("--cohort_csv", type=str, default=None, help="Input ICU cohort CSV")
    parser.add_argument("--cleaned_vitals_jsonl", type=str, default=None, help="Input cleaned vitals JSONL")
    parser.add_argument("--cf_dir", type=str, default=None, help="Directory containing per-vital counterfactual dictionaries")
    parser.add_argument("--original_output", type=str, default=None, help="Output JSONL for original templates")
    parser.add_argument("--counterfactual_output", type=str, default=None, help="Output JSONL for counterfactual templates")
    parser.add_argument("--demographics_counterfactual_output",type=str,default=None,help="Output JSONL for demographic counterfactual templates")
    parser.add_argument("--demographics_seed", type=int, default=42, help="Random seed for demographic age sampling")
    parser.add_argument("--same_age_samples", type=int, default=5, help="Number of same-category age samples")
    parser.add_argument("--cross_age_samples", type=int, default=5, help="Number of per-category cross-category age samples")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    cohort_csv = Path(args.cohort_csv).expanduser() if args.cohort_csv else data_dir / "icu_cohort_data.csv"
    cleaned_vitals_jsonl = Path(args.cleaned_vitals_jsonl).expanduser() if args.cleaned_vitals_jsonl else data_dir / "cleaned_vitals.jsonl"
    cf_dir = Path(args.cf_dir).expanduser() if args.cf_dir else data_dir / "counterfactuals"
    out_original_jsonl = Path(args.original_output).expanduser() if args.original_output else data_dir / "original_notes_template_based.jsonl"
    out_counterfactual_jsonl = Path(args.counterfactual_output).expanduser() if args.counterfactual_output else data_dir / "counterfactual_notes_template_based.jsonl"
    out_demographics_counterfactual_jsonl = (
        Path(args.demographics_counterfactual_output).expanduser()
        if args.demographics_counterfactual_output
        else data_dir / "demographics_counterfactual_notes_template_based.jsonl"
    )

    cf_paths = {
        "oxygen_saturation": cf_dir / "oxygen_saturation_counterfactuals.jsonl",
        "blood_pressure": cf_dir / "blood_pressure_counterfactuals.jsonl",
        "temperature": cf_dir / "temperature_counterfactuals.jsonl",
        "respiration_rate": cf_dir / "respiration_rate_counterfactuals.jsonl",
        "heart_rate": cf_dir / "heart_rate_counterfactuals.jsonl",
    }

    print("Loading cohort demographics...")
    demo_data = load_demo_data(cohort_csv)

    print("Loading cleaned vitals...")
    vitals_data = load_cleaned_vitals(cleaned_vitals_jsonl)

    print("Building original template-based notes...")
    original_templates = build_original_templates(demo_data, vitals_data)
    write_jsonl(out_original_jsonl, original_templates)
    print(f"Saved {len(original_templates):,} original templates to {out_original_jsonl}")

    print("Building counterfactual template-based notes...")
    counterfactual_templates = build_counterfactual_templates(original_templates, cf_paths)
    write_jsonl(out_counterfactual_jsonl, counterfactual_templates)
    print(f"Saved {len(counterfactual_templates):,} counterfactual templates to {out_counterfactual_jsonl}")

    print("Building demographic counterfactual template-based notes...")
    demographic_counterfactual_templates = build_demographic_counterfactual_templates(
        original_templates,
        seed=args.demographics_seed,
        n_same_age_samples=args.same_age_samples,
        n_cross_age_samples=args.cross_age_samples,
    )
    write_jsonl(out_demographics_counterfactual_jsonl, demographic_counterfactual_templates)
    print(
        "Saved "
        f"{len(demographic_counterfactual_templates):,} demographic counterfactual templates "
        f"to {out_demographics_counterfactual_jsonl}"
    )


if __name__ == "__main__":
    main()
