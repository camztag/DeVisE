#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

from section1_utils import (
    DEFAULT_DATA_DIR,
    get_class_severity,
    load_jsonl,
    normalize_class_name,
    write_id_list,
)

suffix_map = {
    "oxygen_saturation": "os",
    "blood_pressure": "bp",
    "temperature": "tp",
    "respiration_rate": "rr",
    "heart_rate": "hr",
}

ALL_VITALS = [
    "oxygen_saturation",
    "blood_pressure",
    "temperature",
    "respiration_rate",
    "heart_rate",
]


def extract_main_number(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value)
    if value.lower() == "nan":
        return None
    nums = re.findall(r"\d+\.?\d*", value)
    return nums[0] if nums else None


def get_vital_code(vital_type: str) -> str:
    return vital_type[:2]


# ============================================================
# SECTION EXTRACTION
# safer: ONLY operate inside PHYSICAL EXAM
# ============================================================

_PHYSICAL_EXAM_RE = re.compile(
    r"(PHYSICAL EXAM:\s*)(.*?)(?=\n\n(?:FAMILY HISTORY:|SOCIAL HISTORY:|MEDICATION ON ADMISSION:|ALLERGIES:|$))",
    flags=re.IGNORECASE | re.DOTALL,
)

def extract_physical_exam_span(text: str):
    """
    Returns:
      {
        "header_start": int,
        "header_end": int,
        "content_start": int,
        "content_end": int,
        "content": str
      }
    or None if PHYSICAL EXAM is not found.
    """
    m = _PHYSICAL_EXAM_RE.search(text)
    if not m:
        return None

    return {
        "header_start": m.start(1),
        "header_end": m.end(1),
        "content_start": m.start(2),
        "content_end": m.end(2),
        "content": m.group(2),
    }


# ============================================================
# REPEATED OXYGEN SATURATION LOGIC
# ============================================================

def compute_repeated_oxsat_map(vitals_data: List[Dict[str, Any]]) -> Dict[int, int]:
    repeated_info = {}

    for entry in vitals_data:
        hadm_id = int(entry["hadm_id"])

        vitals = entry.get("vitals", {})
        temp = vitals.get("temperature", "NaN")
        hr = vitals.get("heart_rate", "NaN")
        ox_sat = vitals.get("oxygen_saturation", "NaN")

        oxsat_val = extract_main_number(ox_sat)
        hr_val = extract_main_number(hr)
        temp_val = extract_main_number(temp)

        if not oxsat_val:
            continue

        repeat_count = 0

        if hr_val and oxsat_val == hr_val:
            repeat_count += 1

        if temp_val and "." not in temp_val and oxsat_val == temp_val:
            repeat_count += 1

        if repeat_count > 0:
            repeated_info[hadm_id] = repeat_count

    return repeated_info


# ============================================================
# LOOKUPS are built inside main()
# ============================================================

def get_priority_patterns(vital_type: str, original_value: Any) -> List[str]:
    base_val = str(original_value).strip("%")
    val = re.escape(base_val)

    if "." in base_val:
        check_val = val
    else:
        check_val = val + r"(?!\.\d)"

    if vital_type == "oxygen_saturation":
        patterns = [
            rf"()({check_val}%RA)\b",
            rf"()({check_val}%)\b",
            rf"(SO2\s*[:=]?\s*)({check_val})(?:on\s+RA)\b",
            rf"()({check_val}SO2Sat)\b",
            rf"(SpO2\s*[:=]?\s*)({check_val}2-3L)\b",
            rf"(O2\s*[:=]?\s*)({check_val})RA\b",
            rf"(O2\s*[:=]?\s*)({check_val})\b",
            rf"()({check_val})/RA\b",
            rf"()({check_val})/2L\b",
            rf"(O2Sats\s*[:=]?\s*)({check_val})RA\b",
            rf"(sat\s*[:=]?\s*)({check_val})RA\b",
            rf"(sat\s*[:=]?\s*)({check_val})\b",
            rf"(O2Sats\s*%?)({check_val})(?:s|'s)?\b",
            rf"(SaO2\s*[:=]?\s*%?)({check_val})(?:s|'s)?\b",
            rf"(spo2\s*[:=]?\s*%?)({check_val})(?:s|'s)?\b",
            rf"(sat(?:uration)?\s*[:=]?\s*%?)({check_val})(?:s|'s)?\b",
            rf"(?<!\/)({check_val})(?!\/)\b",
            rf"()({check_val}\s*RA)\b",
        ]

        if "-" in base_val:
            parts = base_val.split("-")
            if len(parts) == 2:
                left, right = parts[0].strip(), parts[1].strip()
                patterns.insert(0, rf"()({re.escape(left)}\s*-\s*{re.escape(right)}%?)\b")
                patterns.insert(0, rf"()({re.escape(left)}%?->\s*{re.escape(right)}%?)\b")
        return patterns

    elif vital_type == "blood_pressure":
        if "/" in base_val:
            systolic, diastolic = base_val.split("/")
            systolic = systolic.strip()
            diastolic = diastolic.strip()

            if "-" in systolic or "-" in diastolic:
                systolic_parts = systolic.split("-")
                diastolic_parts = diastolic.split("-")
                if len(systolic_parts) == 2 and len(diastolic_parts) == 2:
                    lower_sys = systolic_parts[0].strip()
                    upper_sys = systolic_parts[1].strip()
                    lower_dia = diastolic_parts[0].strip()
                    upper_dia = diastolic_parts[1].strip()
                    systolic_pattern = rf"{re.escape(lower_sys)}s?[-]{re.escape(upper_sys)}s?"
                    diastolic_pattern = rf"{re.escape(lower_dia)}s?[-]{re.escape(upper_dia)}s?"
                    return [rf"(?:BP\s*[:=]?\s*)?({systolic_pattern})\s*/\s*({diastolic_pattern})\b"]

            systolic_escaped = re.escape(systolic)
            diastolic_escaped = re.escape(diastolic)
            if "." not in systolic:
                systolic_escaped += r"(?!\.\d)"
            if "." not in diastolic:
                diastolic_escaped += r"(?!\.\d)"
            return [rf"(?:BP\s*[:=]?\s*)?({systolic_escaped})(?:s|'s)?\s*[-/]\s*({diastolic_escaped})(?:s|'s)?\w*\b"]

        return [rf"(?:BP\s*[:=]?\s*)?({check_val})(?:s|'s)?\w*\b"]

    elif vital_type == "temperature":
        patterns = [
            rf"(T\s*[:=]?\s*)({check_val})(?!%)(?:s|'s)?(?:F|°C|°F)?\b",
            rf"()({check_val})(?!%)(?:s|'s)?(?:F|°C|°F)?\b",
            rf"(Tmax\s*[:=]?\s*)({check_val})(?!%)\b",
            rf"()({check_val})(?!%)%-({check_val})(?!%)%?",
            rf"(Range\s*=\s*)\d{{2}}-\d{{2}}(?:\.\d+)?",
            rf"()({check_val})(?!%)\s*/\s*\d{{2,3}}(?:\.\d+)?",
            rf"\(()({check_val})(?!%)(?:F|°C|°F)?\)",
            rf"(T\s*[:=]?\s*)({check_val})(?!%)(?:C|°C|F|°F|PO)?\b",
            rf"()({check_val})(?!%)(?:C|°C|F|°F|PO)?\b",
            rf"(?<!\/)({check_val})(?!\/)(?!%)\b",
        ]
        patterns.insert(0, rf"(T\s*[:=]?\s*)({check_val})(?!%)(?:ax|PO)\b")

        if "(" in base_val and ")" in base_val:
            patterns.insert(0, rf"()({re.escape(base_val)})\b")

        if "/" in base_val and not base_val.startswith("("):
            lower, upper = base_val.split("/")
            lower = lower.strip()
            upper = upper.strip()
            patterns.insert(0, rf"()({re.escape(lower)})(?!%)\s*/\s*({re.escape(upper)})(?!%)\b")

        return patterns

    elif vital_type == "heart_rate":
        if "-" in base_val:
            lower, upper = base_val.split("-")
            lower = lower.strip()
            upper = upper.strip()
            return [
                rf"(HR\s*[:=]?\s*)(({re.escape(lower)}s?-\s*{re.escape(upper)}s?))\b",
                rf"()(({re.escape(lower)}s?-\s*{re.escape(upper)}s?))\b",
                rf"(Pulse\s*[:=]?\s*)({re.escape(upper)})(?:s|'s)?[A-Za-z]*\b",
            ]

        return [
            rf"(?:P\s*[:=]?\s*)({check_val})(?!\/)(?!%)(?:s|'s)?\b",
            rf"(HR\s*[:=]?\s*)({check_val})reg\b",
            rf"(HR\s*[:=]?\s*)({check_val})(?:\s?ST)\b",
            rf"(P\s*[:=]?\s*)({check_val})reg\b",
            rf"()({check_val})BPM\b",
            rf"(HR\s*[:=]?\s*)({check_val})SR\b",
            rf"(Heart\s*Rate\s*[:=]?\s*|HR\s*[:=]?\s*)({check_val})(?!\/)(?!%)\w*\b",
            rf"(HR\s*[:=]?\s*)({check_val})(?!\/)(?!%)(?:s|'s)?[A-Za-z]*\b",
            rf"(Pulse\s*[:=]?\s*)({check_val})(?!\/)(?!%)(?:s|'s)?[A-Za-z]*\b",
            rf"(P)({check_val})(?!\/)(?!%)(?:s|'s)?[A-Za-z]*\b",
            rf"(?<!\/)({check_val})(?!\/)(?!%)(?:s|'s)?[A-Za-z]*\b",
            rf"\b({check_val})\/min\b",
            rf"(?<!\/)({check_val})(?!\/)(?!%)\b",
        ]

    elif vital_type == "respiration_rate":
        return [
            rf"\b({check_val})_+\b",
            rf"(RR\s*[:=]?\s*)({check_val})(?!\/)(?:s|'s)?\b",
            rf"(R\s*[:=]?\s*)({check_val})(?:s|'s)?\b",
            rf"\b({check_val})\/min\b",
            rf"\bR({check_val})(?!\/)(?:s|'s)?\b",
            rf"\b({check_val})(?!\/)(?:s|'s)?R{{1,2}}\b",
            rf"RR=?\s*({check_val})(?!\/)\b",
            rf"\b({check_val})x\d+\b",
            rf"(?<!HR\s)(?<!HR)\b({check_val})(?!\/)\b",
        ]

    return []


# ============================================================
# MATCHING HELPERS
# ============================================================

class ProxyMatch:
    def __init__(self, real_match):
        self.real_match = real_match

    def start(self):
        return self.real_match.start()

    def end(self):
        return self.real_match.end()

    def group(self, n=0):
        return self.real_match.group(n)

    def groups(self):
        return self.real_match.groups()

    @property
    def lastindex(self):
        return self.real_match.lastindex


def find_best_match(
    section: str,
    vital_type: str,
    original_value: Any,
    hadm_id: int,
    repeated_oxsat: Dict[int, int],
):
    patterns = get_priority_patterns(vital_type, original_value)

    if vital_type == "oxygen_saturation":
        strong_patterns = [p for p in patterns if "%" in p or "RA" in p]
        weak_patterns = [p for p in patterns if p not in strong_patterns]
        ordered_patterns = strong_patterns + weak_patterns

        all_matches = []
        for pat in ordered_patterns:
            for m in re.finditer(pat, section, flags=re.IGNORECASE):
                all_matches.append({
                    "start": m.start(),
                    "end": m.end(),
                    "pattern": pat,
                })

        all_matches.sort(key=lambda x: x["start"])
        desired_index = repeated_oxsat.get(hadm_id, 0)

        if desired_index < len(all_matches):
            selected = all_matches[desired_index]
            regex = re.compile(selected["pattern"], flags=re.IGNORECASE)
            actual_match = regex.search(section, pos=selected["start"], endpos=selected["end"])
            if actual_match is not None:
                return ProxyMatch(actual_match)
        return None

    for pat in patterns:
        candidate = re.search(pat, section, flags=re.IGNORECASE)
        if not candidate:
            continue

        if vital_type == "heart_rate":
            start_idx = candidate.start()
            context_before = section[max(0, start_idx - 10):start_idx].lower()
            if "t:" in context_before or "t =" in context_before:
                continue

        return candidate

    return None


# ============================================================
# REPORT TRACKERS
# ============================================================

already_reported = set()
matched_hadm_ids = set()
unmatched_oxygen = set()


# ============================================================
# MAIN GENERATION
# ============================================================

def main():
    global already_reported, matched_hadm_ids, unmatched_oxygen

    parser = argparse.ArgumentParser(description="Create note-level counterfactuals by swapping vitals inside PHYSICAL EXAM")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Base directory for Section 1 data")
    parser.add_argument("--notes_jsonl", type=str, default=None, help="Input notes JSONL")
    parser.add_argument("--vitals_jsonl", type=str, default=None, help="Input cleaned vitals JSONL")
    parser.add_argument("--cf_dir", type=str, default=None, help="Directory containing per-vital counterfactual dictionaries")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for generated note counterfactuals and reports")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    notes_path = Path(args.notes_jsonl).expanduser() if args.notes_jsonl else data_dir / "icu_notes_admission_clean.jsonl"
    vitals_path = Path(args.vitals_jsonl).expanduser() if args.vitals_jsonl else data_dir / "cleaned_vitals.jsonl"
    cf_dir = Path(args.cf_dir).expanduser() if args.cf_dir else data_dir / "counterfactuals"
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else cf_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cf_paths = {
        "oxygen_saturation": cf_dir / "oxygen_saturation_counterfactuals.jsonl",
        "blood_pressure": cf_dir / "blood_pressure_counterfactuals.jsonl",
        "temperature": cf_dir / "temperature_counterfactuals.jsonl",
        "respiration_rate": cf_dir / "respiration_rate_counterfactuals.jsonl",
        "heart_rate": cf_dir / "heart_rate_counterfactuals.jsonl",
    }

    notes = load_jsonl(notes_path)
    vitals_data = load_jsonl(vitals_path)
    missing_raw_vitals = [
        (entry.get("subject_id"), entry.get("hadm_id"))
        for entry in vitals_data
        if "raw_vitals" not in entry
    ]
    if missing_raw_vitals:
        first_subject_id, first_hadm_id = missing_raw_vitals[0]
        raise ValueError(
            "Input vitals JSONL must include raw_vitals for counterfactual note generation. "
            "Re-run 4-clean-vitals.py with --keep_raw. "
            f"First missing row: subject_id={first_subject_id}, hadm_id={first_hadm_id}"
        )

    vitals_lookup = {
        (entry["subject_id"], entry["hadm_id"]): {
            "cleaned_vitals": entry.get("vitals", {}) or {},
            "raw_vitals": entry.get("raw_vitals", {}) or {},
        }
        for entry in vitals_data
    }
    repeated_oxsat = compute_repeated_oxsat_map(vitals_data)
    already_reported = set()
    matched_hadm_ids = set()
    unmatched_oxygen = set()

    for vital_type, cf_path in cf_paths.items():
        counterfactuals = load_jsonl(cf_path)
        cf_lookup = {entry["hadm_id"]: entry for entry in counterfactuals}
        cf_notes = []

        for note in notes:
            subject_id = note["subject_id"]
            hadm_id = note["hadm_id"]
            key = (subject_id, hadm_id)

            if key not in vitals_lookup or hadm_id not in cf_lookup:
                continue

            vitals_source = vitals_lookup.get(key) or {}
            cleaned_vitals = vitals_source.get("cleaned_vitals", {})
            raw_vitals = vitals_source.get("raw_vitals", {})

            cleaned_value = cleaned_vitals.get(vital_type)
            original_value = raw_vitals.get(vital_type)

            if not original_value or original_value == "NaN":
                continue

            text = note["text"]
            cf_entry = cf_lookup[hadm_id]
            augmentations = cf_entry["augmentations"]

            original_class = cf_entry.get("original", {}).get("original_class")
            original_severity = get_class_severity(vital_type, original_class) if original_class is not None else None

            exam_span = extract_physical_exam_span(text)
            if not exam_span:
                already_reported.add(hadm_id)
                if vital_type == "oxygen_saturation":
                    unmatched_oxygen.add(hadm_id)
                continue

            section = exam_span["content"]
            section_start = exam_span["content_start"]

            match = find_best_match(section, vital_type, original_value, hadm_id, repeated_oxsat)

            if not match:
                already_reported.add(hadm_id)
                if vital_type == "oxygen_saturation":
                    unmatched_oxygen.add(hadm_id)
                continue

            matched_hadm_ids.add(hadm_id)

            for aug in augmentations:
                counterfactual_class = aug.get("new_class")
                counterfactual_severity = get_class_severity(vital_type, counterfactual_class) if counterfactual_class is not None else None

                if original_severity is not None and counterfactual_severity is not None:
                    class_diff = counterfactual_severity - original_severity
                    class_diff_abs = abs(counterfactual_severity) - abs(original_severity)
                else:
                    class_diff = None
                    class_diff_abs = None

                for aug_key, new_value in aug.items():
                    if "new_value" not in aug_key:
                        continue

                    full_start = section_start + match.start()
                    full_end = section_start + match.end()
                    full_match_text = match.group(0)

                    if vital_type == "heart_rate" and "-" in str(original_value):
                        new_text = str(new_value)
                    else:
                        if vital_type == "blood_pressure":
                            label_match = re.match(r"^(.*?BP\s*[:=]?\s*)", full_match_text, flags=re.IGNORECASE)
                            prefix = label_match.group(1) if label_match else ""
                            suffix = ""
                        elif vital_type == "respiration_rate" and "x" in full_match_text:
                            parts = full_match_text.split("x", 1)
                            prefix = ""
                            suffix = "x" + parts[1] if len(parts) > 1 else ""
                        else:
                            prefix = match.group(1) if match.lastindex and match.lastindex >= 2 else ""
                            suffix = ""

                        if vital_type == "oxygen_saturation":
                            new_text = prefix + str(new_value) + "%" + suffix
                        else:
                            new_text = prefix + str(new_value) + suffix

                    safe_class = str(counterfactual_class).lower().replace(" ", "_").replace("/", "_")
                    new_note = note.copy()
                    new_note["text"] = text[:full_start] + new_text + text[full_end:]
                    new_note["id"] = f"{hadm_id}_{vital_type[:2]}_{safe_class}_{aug_key.split('_')[-1]}"
                    new_note["original_class"] = normalize_class_name(original_class) if original_class is not None else None
                    new_note["original_severity"] = original_severity
                    new_note["counterfactual_class"] = normalize_class_name(counterfactual_class) if counterfactual_class is not None else None
                    new_note["counterfactual_severity"] = counterfactual_severity
                    new_note["class_diff"] = class_diff
                    new_note["class_diff_abs"] = class_diff_abs

                    cf_notes.append(new_note)

        out_path = output_dir / f"cf_{suffix_map[vital_type]}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for note in cf_notes:
                f.write(json.dumps(note, ensure_ascii=False) + "\n")
        print(f"Saved {len(cf_notes)} notes for {vital_type} to {out_path}")

    not_found_hadm_ids = sorted(already_reported)
    not_found_path = output_dir / "not_found_hadm_ids.txt"
    write_id_list(not_found_path, not_found_hadm_ids)
    print(f"Saved list of {len(not_found_hadm_ids)} hadm_ids with at least one unmatched vital to {not_found_path}")

    eligible_hadm_ids = set()
    for (_, hadm_id), vitals_payload in vitals_lookup.items():
        vitals = vitals_payload.get("cleaned_vitals", {})
        for vital in ALL_VITALS:
            if vitals.get(vital) not in [None, "NaN", ""]:
                eligible_hadm_ids.add(hadm_id)
                break

    none_matched = sorted(eligible_hadm_ids - matched_hadm_ids)
    none_matched_path = output_dir / "none_matched_vitals.txt"
    write_id_list(none_matched_path, none_matched)
    print(f"Saved list of {len(none_matched)} hadm_ids with no vital matches at all to {none_matched_path}")

    oxygen_not_matched = sorted(unmatched_oxygen)
    oxygen_not_matched_path = output_dir / "oxygen_saturation_not_matched.txt"
    write_id_list(oxygen_not_matched_path, oxygen_not_matched)
    print(f"Saved list of {len(oxygen_not_matched)} hadm_ids with oxygen saturation unmatched to {oxygen_not_matched_path}")


if __name__ == "__main__":
    main()
