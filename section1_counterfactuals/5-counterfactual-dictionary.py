#!/usr/bin/env python3

import os
import json
import random
import argparse
from collections import defaultdict, Counter
from typing import Optional, Any, Dict, List

import numpy as np


# ============================================================
# RANGES
# ============================================================

vital_sampling_ranges = {
    "heart_rate": {
        "Very low/LTL": (1, 40),
        "Low": (41, 50),
        "Normal": (51, 90),
        "High": (91, 110),
        "Very high/LTH": (111, 200),
    },
    "respiration_rate": {
        "Very low/LTL": (1, 8),
        "Low": (9, 11),
        "Normal": (12, 20),
        "High": (21, 24),
        "Very high/LTH": (25, 50),
    },
    "oxygen_saturation": {
        "Very low/LTL": (1, 93),
        "Low": (94, 95),
        "Normal": (96, 100),
    },
    "temperature": {
        "Very low/LTL": (70.0, 89.4),
        "Low": (89.5, 94.9),
        "Normal": (95.0, 100.2),
        "High": (100.3, 103.9),
        "Very high/LTH": (104.0, 110.0),
    },
}

bp_systolic_ranges = {
    "Very low/LTL": (1, 70),
    "Low": (71, 89),
    "Normal": (90, 119),
    "High": (120, 139),
    "Very high/LTH": (140, 220),
}

bp_diastolic_ranges = {
    "Very low/LTL": (1, 40),
    "Low": (41, 59),
    "Normal": (60, 79),
    "High": (80, 89),
    "Very high/LTH": (90, 140),
}

severity_map = {
    "Very low/LTL": -2,
    "Low": -1,
    "Normal": 0,
    "High": 1,
    "Very high/LTH": 2,
}

ALL_VITALS = list(vital_sampling_ranges.keys()) + ["blood_pressure"]


# ============================================================
# HELPERS
# ============================================================

def parse_clean_numeric(value: Any) -> Optional[float]:
    """
    Parse already-clean non-BP values coming from script 1.
    Expected examples:
    - "98"
    - "98.6"
    - 98
    - 98.6
    - "NaN" -> None
    """
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if value == "" or value == "NaN":
            return None

    try:
        return float(value)
    except Exception:
        return None


def parse_clean_bp(bp_value: Any) -> Optional[str]:
    """
    Accept only already-clean blood pressure values from script 1.
    Expected format: 'SYS/DIA'
    """
    if bp_value is None:
        return None

    bp_value = str(bp_value).strip()
    if bp_value == "" or bp_value == "NaN":
        return None

    if "/" not in bp_value:
        return None

    parts = bp_value.split("/")
    if len(parts) != 2:
        return None

    try:
        systolic = int(float(parts[0]))
        diastolic = int(float(parts[1]))
    except Exception:
        return None

    return f"{systolic}/{diastolic}"


def classify_bp_value(bp_value: str) -> Optional[str]:
    if not bp_value or "/" not in bp_value:
        return None

    try:
        systolic, diastolic = map(int, bp_value.split("/"))
    except Exception:
        return None

    def classify_s(s: int) -> str:
        if s <= 70:
            return "Very low/LTL"
        if 71 <= s <= 89:
            return "Low"
        if 90 <= s <= 119:
            return "Normal"
        if 120 <= s <= 139:
            return "High"
        return "Very high/LTH"

    def classify_d(d: int) -> str:
        if d <= 40:
            return "Very low/LTL"
        if 41 <= d <= 59:
            return "Low"
        if 60 <= d <= 79:
            return "Normal"
        if 80 <= d <= 89:
            return "High"
        return "Very high/LTH"

    sys_class = classify_s(systolic)
    dia_class = classify_d(diastolic)

    return sys_class if abs(severity_map[sys_class]) >= abs(severity_map[dia_class]) else dia_class


def classify_non_bp_value(vital: str, value: Any) -> Optional[Dict[str, Any]]:
    if vital not in vital_sampling_ranges:
        return None

    num_val = parse_clean_numeric(value)
    if num_val is None:
        return None

    for cls_name, (low, high) in vital_sampling_ranges[vital].items():
        if low <= float(num_val) <= high:
            if vital == "temperature":
                cleaned_value = round(float(num_val), 1)
            else:
                cleaned_value = int(round(float(num_val)))
            return {"value": cleaned_value, "class": cls_name}

    return None


def load_test_hadm_ids(path: Optional[str]) -> Optional[set]:
    if path is None:
        return None

    hadm_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # support plain IDs or json/jsonl objects containing hadm_id
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "hadm_id" in obj:
                    hadm_ids.add(int(obj["hadm_id"]))
                else:
                    hadm_ids.add(int(obj))
            except Exception:
                hadm_ids.add(int(line))

    return hadm_ids


# ============================================================
# LOAD + CLASSIFY PATIENT RECORDS
# ============================================================

def build_patient_records(input_path: str, test_hadm_ids: Optional[set] = None) -> List[Dict[str, Any]]:
    patient_records = []

    with open(input_path, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSON at line {line_num}: {e}")
                continue

            hadm_id = data.get("hadm_id")
            subject_id = data.get("subject_id")
            vitals = data.get("vitals", {})

            if hadm_id is None:
                continue

            try:
                hadm_id = int(hadm_id)
            except Exception:
                continue

            if test_hadm_ids is not None and hadm_id not in test_hadm_ids:
                continue

            record = {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
            }

            for vital, value in vitals.items():
                if value == "NaN":
                    continue

                # BLOOD PRESSURE
                if vital == "blood_pressure":
                    bp_clean = parse_clean_bp(value)

                    if bp_clean is None:
                        continue

                    final_class = classify_bp_value(bp_clean)
                    if final_class is None:
                        continue

                    record[vital] = {
                        "value": bp_clean,
                        "class": final_class
                    }
                    continue

                # OTHER VITALS
                result = classify_non_bp_value(vital, value)
                if result is not None:
                    record[vital] = result

            patient_records.append(record)

    print(f"Kept {len(patient_records)} patient records from the vitals file")
    return patient_records


# ============================================================
# COUNTERFACTUAL GENERATION
# ============================================================

def sample_non_bp_values(vital: str, cls_name: str, n_samples: int) -> List[Any]:
    low, high = vital_sampling_ranges[vital][cls_name]

    if vital != "temperature":
        possible_values = list(range(int(low), int(high) + 1))
    else:
        possible_values = [round(x, 1) for x in np.arange(low, high + 0.1, 0.1)]

    random.shuffle(possible_values)
    return possible_values[: min(n_samples, len(possible_values))]


def sample_bp_values(cls_name: str, n_samples: int, max_attempts: int = 100) -> List[str]:
    sys_low, sys_high = bp_systolic_ranges[cls_name]
    dia_low, dia_high = bp_diastolic_ranges[cls_name]

    possible_sys = list(range(sys_low, sys_high + 1))
    possible_dia = list(range(dia_low, dia_high + 1))

    sampled = []
    attempts = 0

    while len(sampled) < n_samples and attempts < max_attempts:
        s = random.choice(possible_sys)
        d = random.choice(possible_dia)
        bp = f"{s}/{d}"
        if bp not in sampled:
            sampled.append(bp)
        attempts += 1

    return sampled


def generate_counterfactuals(
    patient_records: List[Dict[str, Any]],
    output_dir: str,
    n_samples_per_class: int = 5,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for vital in ALL_VITALS:
        output = []

        for record in patient_records:
            if vital not in record:
                continue

            orig_value = record[vital]["value"]
            orig_class = record[vital]["class"]
            hadm_id = record["hadm_id"]
            subject_id = record["subject_id"]

            augmentations = []
            counter = 1

            if vital != "blood_pressure":
                for cls_name in vital_sampling_ranges[vital].keys():
                    sampled = sample_non_bp_values(vital, cls_name, n_samples_per_class)

                    for val in sampled:
                        augmentations.append({
                            "new_class": cls_name,
                            f"new_value_{counter}": val
                        })
                        counter += 1

            else:
                for cls_name in bp_systolic_ranges.keys():
                    sampled = sample_bp_values(cls_name, n_samples_per_class)

                    for val in sampled:
                        augmentations.append({
                            "new_class": cls_name,
                            f"new_value_{counter}": val
                        })
                        counter += 1

            output.append({
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "original": {
                    "original_class": orig_class,
                    "original_value": orig_value
                },
                "augmentations": augmentations
            })

        output_file = os.path.join(output_dir, f"{vital}_counterfactuals.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in output:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("counterfactuals created per vital")


# ============================================================
# CLASS COUNT SUMMARY
# ============================================================

def summarize_class_counts(patient_records: List[Dict[str, Any]], output_dir: str) -> None:
    original_class_counts = defaultdict(Counter)
    counterfactual_class_counts = defaultdict(Counter)

    # Original counts
    for record in patient_records:
        for vital in record:
            if vital in ["subject_id", "hadm_id"]:
                continue
            original_class_counts[vital][record[vital]["class"]] += 1

    # Counterfactual counts
    for vital in ALL_VITALS:
        filepath = os.path.join(output_dir, f"{vital}_counterfactuals.jsonl")
        if not os.path.exists(filepath):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                for aug in data["augmentations"]:
                    cls = aug["new_class"]
                    counterfactual_class_counts[vital][cls] += 1

    print("\n===== ORIGINAL CLASS COUNTS =====")
    for vital, counts in original_class_counts.items():
        print(f"\nVital: {vital}")
        for cls, n in counts.items():
            print(f"  {cls}: {n}")

    print("\n===== COUNTERFACTUAL CLASS COUNTS =====")
    for vital, counts in counterfactual_class_counts.items():
        print(f"\nVital: {vital}")
        for cls, n in counts.items():
            print(f"  {cls}: {n}")

    print("\nClass counts computed.\n")


# ============================================================
# DEBUG / COVERAGE
# ============================================================

def print_vital_coverage(patient_records: List[Dict[str, Any]]) -> None:
    print("\n===== VITAL COVERAGE =====")
    print(sum("oxygen_saturation" in r for r in patient_records), "patients have oxygen_saturation")

    for vital in vital_sampling_ranges.keys():
        count = sum(vital in r for r in patient_records)
        print(vital, count)

    bp_count = sum("blood_pressure" in r for r in patient_records)
    print("blood_pressure", bp_count)
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Classify cleaned vitals and generate counterfactuals")
    parser.add_argument("--input_file", required=True, help="Cleaned vitals JSONL")
    parser.add_argument("--output_dir", required=True, help="Directory to save per-vital counterfactual JSONLs")
    parser.add_argument("--test_hadm_ids_file", default=None, help="Optional file with hadm_ids to keep")
    parser.add_argument("--n_samples_per_class", type=int, default=5, help="How many counterfactual values to sample per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    test_hadm_ids = load_test_hadm_ids(args.test_hadm_ids_file)

    patient_records = build_patient_records(
        input_path=args.input_file,
        test_hadm_ids=test_hadm_ids,
    )

    generate_counterfactuals(
        patient_records=patient_records,
        output_dir=args.output_dir,
        n_samples_per_class=args.n_samples_per_class,
    )

    summarize_class_counts(
        patient_records=patient_records,
        output_dir=args.output_dir,
    )

    print_vital_coverage(patient_records)


if __name__ == "__main__":
    main()