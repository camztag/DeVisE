#!/usr/bin/env python3

import argparse
import json
import difflib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

from section1_utils import DEFAULT_DATA_DIR, load_jsonl, write_id_list, write_jsonl

# Goals:
# 1. Identify and report duplicate vital sign values within the dataset.
# 2. Highlight differences between original and counterfactual texts.
# 3. Ensure comprehensive coverage of all relevant vital signs.
# 4. Merge all CF files, generate diff report comparing original vs CF notes, flag any problematic CFs (e.g. with 0 or >1 changes, or changes outside physical exam), and generate summary stats about the dataset and checks.

# inputs: - original notes JSONL (hadm_id, text)
#         - original vitals JSONL (hadm_id, vitals{vital_name: value})
#         - CF JSONL files (id, hadm_id, text, original_class, counterfactual_class, original_severity, counterfactual_severity, class_diff, class_diff_abs)
# outputs: - merged CF JSONL file
#          - diff report JSONL file (hadm_id, counterfactual_id, n_changes, changes[{tag, out, in, section, original_range, cf_range}], original_class, counterfactual_class, original_severity, counterfactual_severity, class_diff, class_diff_abs)
#          - problematic CF JSONL file (hadm_id, counterfactual_id, problems[n_changes_0, n_changes_>1, change_outside_physical_exam], n_changes, changes[{tag, out, in, section, original_range, cf_range}])           

# ============================================================
# DUPLICATE VITAL VALUE REPORT
# ============================================================

def normalize_value(val: Any) -> Optional[float]:
    if val is None:
        return None

    val = str(val).strip()
    if val == "" or val == "NaN":
        return None

    val = val.replace("%", "")

    try:
        return float(val)
    except ValueError:
        return None


def find_duplicate_original_vitals(vitals_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    duplicates = []

    for line_number, row in enumerate(vitals_rows, start=1):
        vitals = row.get("vitals", {})
        values_seen = {}
        duplicate_values = []

        for vital_name, val in vitals.items():
            norm_val = normalize_value(val)
            if norm_val is None:
                continue

            if norm_val in values_seen:
                duplicate_values.append({
                    "value": norm_val,
                    "first_vital": values_seen[norm_val],
                    "duplicate_vital": vital_name,
                })
            else:
                values_seen[norm_val] = vital_name

        if duplicate_values:
            duplicates.append({
                "line_number": line_number,
                "subject_id": row.get("subject_id"),
                "hadm_id": row.get("hadm_id"),
                "vitals": vitals,
                "duplicate_values": duplicate_values,
            })

    return duplicates


# ============================================================
# DIFF HELPERS
# ============================================================

def find_token_differences(original_text: str, cf_text: str) -> List[Dict[str, Any]]:
    original_tokens = original_text.split()
    cf_tokens = cf_text.split()
    matcher = difflib.SequenceMatcher(None, original_tokens, cf_tokens)

    differences = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete", "insert"):
            out_chunk = " ".join(original_tokens[i1:i2])
            in_chunk = " ".join(cf_tokens[j1:j2])
            differences.append({
                "tag": tag,
                "out": out_chunk,
                "in": in_chunk,
                "original_range": [i1, i2],
                "cf_range": [j1, j2],
            })
    return differences


def get_physical_exam_token_range(text: str) -> Optional[List[int]]:
    lines = text.split("\n")
    all_tokens = []
    line_token_ranges = []
    current_token_count = 0

    for line in lines:
        tokens = line.split()
        start_idx = current_token_count
        end_idx = start_idx + len(tokens)
        line_token_ranges.append((start_idx, end_idx))
        all_tokens.extend(tokens)
        current_token_count = end_idx

    pe_start = None
    pe_end = None

    for i, line in enumerate(lines):
        line_upper = line.strip().upper()
        if pe_start is None and line_upper.startswith("PHYSICAL EXAM"):
            pe_start = line_token_ranges[i][0]
        elif pe_start is not None and (
            line_upper.startswith("FAMILY HISTORY")
            or line_upper.startswith("SOCIAL HISTORY")
            or line_upper.startswith("MEDICATION ON ADMISSION")
            or line_upper.startswith("ALLERGIES")
        ):
            pe_end = line_token_ranges[i][0]
            break

    if pe_start is not None and pe_end is None:
        pe_end = current_token_count

    if pe_start is not None and pe_end is not None:
        return [pe_start, pe_end]

    return None


def is_in_physical_exam_section(token_range: List[int], pe_range: Optional[List[int]]) -> bool:
    if not pe_range:
        return False

    pe_start, pe_end = pe_range
    d_start, d_end = token_range
    return not (d_end <= pe_start or d_start >= pe_end)


# ============================================================
# ID / METADATA HELPERS
# ============================================================

def infer_vital_from_id(cf_id: str) -> Optional[str]:
    parts = str(cf_id).split("_")
    if len(parts) < 2:
        return None

    code = parts[1]
    code_to_vital = {
        "bl": "blood_pressure",
        "he": "heart_rate",
        "re": "respiration_rate",
        "ox": "oxygen_saturation",
        "te": "temperature",
    }
    return code_to_vital.get(code)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run checks over generated counterfactual notes")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Base directory for Section 1 data")
    parser.add_argument("--original_notes_jsonl", type=str, default=None, help="Original clean notes JSONL")
    parser.add_argument("--original_vitals_jsonl", type=str, default=None, help="Original cleaned vitals JSONL")
    parser.add_argument("--cf_dir", type=str, default=None, help="Directory containing generated counterfactual notes")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    original_notes_path = Path(args.original_notes_jsonl).expanduser() if args.original_notes_jsonl else data_dir / "icu_notes_admission_clean.jsonl"
    original_vitals_path = Path(args.original_vitals_jsonl).expanduser() if args.original_vitals_jsonl else data_dir / "cleaned_vitals.jsonl"
    cf_dir = Path(args.cf_dir).expanduser() if args.cf_dir else data_dir / "counterfactuals"
    cf_files = ["cf_bp.jsonl", "cf_hr.jsonl", "cf_rr.jsonl", "cf_os.jsonl", "cf_tp.jsonl"]

    merged_cf_path = cf_dir / "cf_all.jsonl"
    diffs_output_path = cf_dir / "cf_all_diffs.jsonl"
    duplicate_vitals_output_path = cf_dir / "duplicate_original_vitals.jsonl"
    problematic_cf_output_path = cf_dir / "problematic_counterfactuals.jsonl"
    summary_output_path = cf_dir / "cf_summary.json"
    missing_original_notes_path = cf_dir / "missing_original_notes.txt"

    print("Loading original notes...")
    original_rows = load_jsonl(original_notes_path)
    original_notes = {row["hadm_id"]: row["text"] for row in original_rows if "hadm_id" in row and "text" in row}

    print("Loading original vitals...")
    original_vitals_rows = load_jsonl(original_vitals_path)

    print("Checking duplicate original vitals...")
    duplicate_rows = find_duplicate_original_vitals(original_vitals_rows)
    write_jsonl(duplicate_vitals_output_path, duplicate_rows)
    print(f"Saved {len(duplicate_rows)} duplicate-vitals rows to {duplicate_vitals_output_path}")

    print("Loading and merging counterfactual files...")
    merged_rows = []
    per_file_counts = {}
    per_vital_counts = Counter()
    per_cf_class_counts = Counter()
    per_original_class_counts = Counter()

    for cf_filename in cf_files:
        cf_path = cf_dir / cf_filename
        rows = load_jsonl(cf_path)
        per_file_counts[cf_filename] = len(rows)
        merged_rows.extend(rows)

        for row in rows:
            cf_id = row.get("id", "")
            vital = infer_vital_from_id(cf_id)
            if vital is not None:
                per_vital_counts[vital] += 1

            if row.get("counterfactual_class") is not None:
                per_cf_class_counts[row["counterfactual_class"]] += 1

            if row.get("original_class") is not None:
                per_original_class_counts[row["original_class"]] += 1

    write_jsonl(merged_cf_path, merged_rows)
    print(f"Saved merged file with {len(merged_rows)} rows to {merged_cf_path}")

    print("Generating diff report...")
    diff_rows = []
    problematic_rows = []
    missing_original_hadm_ids = set()

    n_missing_original = 0
    n_single_change = 0
    n_multi_change = 0
    n_zero_change = 0
    n_outside_physical_exam = 0
    n_only_physical_exam = 0

    for row in merged_rows:
        hadm_id = row.get("hadm_id")
        cf_id = row.get("id")
        cf_text = row.get("text", "")

        orig_text = original_notes.get(hadm_id)
        if not orig_text:
            n_missing_original += 1
            missing_original_hadm_ids.add(hadm_id)
            problematic_rows.append({
                "hadm_id": hadm_id,
                "counterfactual_id": cf_id,
                "problem": "missing_original_note",
            })
            continue

        diffs = find_token_differences(orig_text, cf_text)
        pe_range = get_physical_exam_token_range(orig_text)

        changes_output = []
        has_outside_pe = False

        for d in diffs:
            section = "PHYSICAL EXAM" if is_in_physical_exam_section(d["original_range"], pe_range) else "OTHER"
            if section == "OTHER":
                has_outside_pe = True

            if d["out"] or d["in"]:
                changes_output.append({
                    "tag": d["tag"],
                    "out": d["out"],
                    "in": d["in"],
                    "section": section,
                    "original_range": d["original_range"],
                    "cf_range": d["cf_range"],
                })

        n_changes = len(changes_output)

        if n_changes == 0:
            n_zero_change += 1
        elif n_changes == 1:
            n_single_change += 1
        else:
            n_multi_change += 1

        if has_outside_pe:
            n_outside_physical_exam += 1
        else:
            n_only_physical_exam += 1

        diff_result = {
            "hadm_id": hadm_id,
            "counterfactual_id": cf_id,
            "vital": infer_vital_from_id(cf_id),
            "n_changes": n_changes,
            "changes": changes_output,
            "original_class": row.get("original_class"),
            "original_severity": row.get("original_severity"),
            "counterfactual_class": row.get("counterfactual_class"),
            "counterfactual_severity": row.get("counterfactual_severity"),
            "class_diff": row.get("class_diff"),
            "class_diff_abs": row.get("class_diff_abs"),
        }
        diff_rows.append(diff_result)

        problem_flags = []
        if n_changes != 1:
            problem_flags.append(f"n_changes_{n_changes}")
        if has_outside_pe:
            problem_flags.append("change_outside_physical_exam")

        if problem_flags:
            problematic_rows.append({
                "hadm_id": hadm_id,
                "counterfactual_id": cf_id,
                "problems": problem_flags,
                "n_changes": n_changes,
                "changes": changes_output,
            })

    write_jsonl(diffs_output_path, diff_rows)
    print(f"Saved {len(diff_rows)} diff rows to {diffs_output_path}")

    write_jsonl(problematic_cf_output_path, problematic_rows)
    print(f"Saved {len(problematic_rows)} problematic rows to {problematic_cf_output_path}")

    write_id_list(missing_original_notes_path, sorted(missing_original_hadm_ids))
    print(f"Saved {len(missing_original_hadm_ids)} missing original hadm_ids to {missing_original_notes_path}")

    print("Saving summary...")
    summary = {
        "n_original_notes": len(original_notes),
        "n_original_vitals_rows": len(original_vitals_rows),
        "n_duplicate_original_vitals_rows": len(duplicate_rows),
        "n_merged_counterfactual_rows": len(merged_rows),
        "per_file_counts": dict(per_file_counts),
        "per_vital_counts": dict(per_vital_counts),
        "original_class_counts": dict(per_original_class_counts),
        "counterfactual_class_counts": dict(per_cf_class_counts),
        "diff_summary": {
            "n_diff_rows": len(diff_rows),
            "n_missing_original": n_missing_original,
            "n_zero_change": n_zero_change,
            "n_single_change": n_single_change,
            "n_multi_change": n_multi_change,
            "n_outside_physical_exam": n_outside_physical_exam,
            "n_only_physical_exam": n_only_physical_exam,
        },
        "n_problematic_counterfactual_rows": len(problematic_rows),
    }

    with open(summary_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved summary to {summary_output_path}")


if __name__ == "__main__":
    main()
