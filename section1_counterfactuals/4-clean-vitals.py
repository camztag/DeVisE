#!/usr/bin/env python3
"""
Clean vitals extracted by the vLLM script.

Expected input JSONL (one per line), e.g.
{
  "subject_id": 10343369,
  "hadm_id": 25921885,
  "vitals": {
    "temperature": "102.2",
    "heart_rate": "110",
    "blood_pressure": "110/69",
    "respiration_rate": "25",
    "oxygen_saturation": "100%"
  }
}

Output JSONL keeps subject_id/hadm_id and writes cleaned vitals + optional raw_vitals.
"""

import argparse
import json
import re
from typing import Any, Dict, Optional


VITAL_KEYS = ["temperature", "heart_rate", "blood_pressure", "respiration_rate", "oxygen_saturation"]


# ----------------------------
# Helpers
# ----------------------------
def clean_numeric_value(value: Optional[str]) -> Optional[Any]:
    """
    Extract/approximate numeric values from messy formats, including:
    - '95%', '95 %'
    - '94RA', '94 RA'
    - '70s'
    - '37.7 (99.9)'
    - '70-79' or '70–79' (midpoint)
    - '90-96/40-45' (BP ranges)
    - '120/70'
    Returns:
      - float for single numeric values
      - 'SYS/DIA' string for blood pressure if detected
      - None if not parseable
    """
    if value is None:
        return None

    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return None

    # CLEAN BP CASES: "90-96/40-45", "120/70"
    if "/" in value:
        parts = value.split("/")
        if len(parts) == 2:
            sys_clean = clean_numeric_value(parts[0])
            dia_clean = clean_numeric_value(parts[1])
            if sys_clean is not None and dia_clean is not None:
                try:
                    return f"{int(round(float(sys_clean)))}/{int(round(float(dia_clean)))}"
                except Exception:
                    return None
            return None

    # Remove percent signs, RA labels
    value = value.replace("%", "")
    value = re.sub(r"[Rr][Aa]\b", "", value)

    # Remove parentheses: "37.7 (99.9)" -> "37.7"
    value = value.split("(")[0].strip()

    # Handle ranges like "70-79" or "90–96"
    if "-" in value or "–" in value:
        parts = re.split(r"[-–]", value)
        nums = []
        for p in parts:
            p = re.sub(r"[^\d.]", "", p)
            if p:
                nums.append(float(p))
        if len(nums) == 2:
            return sum(nums) / 2
        return None

    # Handle "70s" -> 70
    if re.match(r"^\d+s$", value):
        return float(value[:-1])

    # Extract first numeric portion
    m = re.search(r"(\d+(\.\d+)?)", value)
    if m:
        return float(m.group(1))

    return None


def convert_temp_c_to_f(temp_str: str) -> str:
    """
    Convert temperatures that look like Celsius (30 <= C < 50) to Fahrenheit (1 decimal).
    Otherwise return original.
    """
    try:
        c = float(temp_str)
        if 30 <= c < 50:
            f = round((c * 9 / 5) + 32, 1)
            return str(f)
    except ValueError:
        pass
    return temp_str


# ----------------------------
# Special cases
# ----------------------------
def handle_special_cases(vital: str, value: str) -> Optional[str]:
    v = value.strip()
    vl = v.lower()

    # Broad "missing/unmeasured" bucket
    if re.search(
        r"(unmeasured|na\b|unmeasurable|unrecorded|none|unable to obtain|unobtainable|no measurable blood pressure|unavailable)",
        vl,
    ):
        return "NaN"

    if vital == "blood_pressure":
        if "elevated" in vl:
            return "129"
        if vl == "ar 58":
            return "85"
        if "normotensive" in vl or "normotension" in vl:
            return "115/75"
        if "hypotensive" in vl:
            return "90/60"

        # (r) 120/70
        m = re.search(r"\(r\)\s*(\d{2,3}/\d{2,3})", vl)
        if m:
            return m.group(1)

        # SBP 120 / sbp120 / sbp 120
        if re.match(r"^\s*sbp\s*\d{2,3}\b", vl):
            m2 = re.search(r"\d{2,3}", vl)
            return m2.group(0) if m2 else None

        # (140-150)/(60-70) or 143-156/76-79 -> choose first sys and first dia
        if re.search(r"\(?\d{2,3}-\d{2,3}\)?/\(?\d{2,3}-\d{2,3}\)?", vl):
            nums = re.findall(r"\d{2,3}", vl)
            if len(nums) >= 4:
                return f"{nums[0]}/{nums[2]}"

    if vital == "oxygen_saturation":
        if "198" in vl:
            return "98"
        if vl == "low":
            return "94"
        if "oxygenating well" in vl:
            return "99"
        m = re.search(r"\b(?:greater|above|over)\s*(?:than\s*)?(\d{2,3})%?", vl)
        if m:
            return m.group(1)

    if vital == "temperature":
        if re.match(r"t___\.3", v, re.IGNORECASE):
            return "NaN"
        if re.search(
            r"\b(afebrile|afeb|af|normothermic|non febrile|aferbile|normal|aefebrile|afebril|afrebile|afebrie)\b",
            vl,
        ):
            return "99"
        if re.search(r"\bfebrile\b", vl):
            return "101"

    if vital == "respiration_rate":
        if re.search(r"(intubated|no spontaneous respirations)", vl):
            return "NaN"

    return None


# ----------------------------
# General normalization
# ----------------------------
def clean_vital(vital: str, value: Any) -> str:
    """
    Returns a cleaned string for the vital, or "NaN".
    Tries special-case mappings first, then pattern/range cleanup, then numeric extraction.

    - blood_pressure may remain as "SYS/DIA" 
    - blood_pressure may also remain as a single number in cases where the source
      only expresses systolic information (e.g. "elevated" -> "129", "SBP 120" -> "120")
    """
    if value is None:
        return "NaN"

    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return "NaN"

    # Handle explicit "NaN" string (case-sensitive) to preserve intentional NaN labels from extraction step
    if value == "NaN":
        return "NaN"

    special = handle_special_cases(vital, value)
    if special is not None:
        return special

    # Keep already-clean BP like "120/80"
    if vital == "blood_pressure" and re.fullmatch(r"\d{2,3}/\d{2,3}", value):
        return value

    # Keep already-clean temperature like "98.6" or "102.2"
    if vital == "temperature" and re.fullmatch(r"\d{2,3}(\.\d+)?", value):
        # convert C->F if it looks Celsius
        return convert_temp_c_to_f(value)

    # Mid/high/low patterns: "mid 90s" / "low-100"
    m = re.search(r"(mid|high|low)[ -]?(\d{2,3})", value, flags=re.IGNORECASE)
    if m:
        return m.group(2)

    m = re.search(r"low to mid\s*(\d{2,3})", value, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.search(r"above[ -]?(\d{2,3})", value, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # % or %RA etc (for SpO2)
    m = re.match(r"^\s*(\d{1,3})\s*%.*$", value)
    if m:
        return m.group(1)

    # ~XX, >XX, <XX
    m = re.match(r"^\s*[~<>]?\s*(\d{1,3})(?:\.\d+)?\b", value)
    if m and vital != "temperature":
        return m.group(1)

    # 70s -> 70
    if re.match(r"^\d{2,3}s$", value):
        return value[:-1]

    # BP forms like "110/60-70" -> 110/60 ; "118-145/59-75" -> 118/59
    if vital == "blood_pressure":
        if re.match(r"^\d+/\d+-\d+$", value):
            nums = re.findall(r"\d+", value)
            if len(nums) >= 2:
                return f"{nums[0]}/{nums[1]}"
        if re.match(r"^\d+-\d+/\d+-\d+$", value):
            nums = re.findall(r"\d+", value)
            if len(nums) >= 3:
                return f"{nums[0]}/{nums[2]}"
        if re.match(r"^\d+/\s*[-_]+$", value):
            return value.split("/")[0]

        # systolic, SBP, sbp<
        if "systolic" in value.lower() or "sbp" in value.lower():
            m = re.search(r"\d{2,3}", value)
            if m:
                return m.group(0)

    # If value is just .14 -> treat as 14 (x100) (your rule)
    if re.match(r"^\.\d+$", value):
        try:
            return str(int(float(value) * 100))
        except Exception:
            return "NaN"

    # Final fallback: parse numeric/range midpoint logic
    parsed = clean_numeric_value(value)
    if parsed is None:
        return "NaN"

    if vital == "blood_pressure" and isinstance(parsed, str) and re.fullmatch(r"\d{2,3}/\d{2,3}", parsed):
        return parsed

    # For oxygen saturation: prefer integer percent
    if vital == "oxygen_saturation":
        try:
            n = float(parsed)
            return str(int(round(n)))
        except Exception:
            return "NaN"

    if vital == "temperature":
        # parsed numeric might be Celsius -> convert if needed
        try:
            t = float(parsed)
            out = f"{t:.1f}".rstrip("0").rstrip(".")
            return convert_temp_c_to_f(out)
        except Exception:
            return "NaN"

    # For the rest, return int if whole number else keep as clean string
    try:
        n = float(parsed)
        if abs(n - round(n)) < 1e-9:
            return str(int(round(n)))
        return str(n)
    except Exception:
        return "NaN"


# ----------------------------
# File processing
# ----------------------------
def clean_vitals_file(
    input_path: str,
    output_path: str,
    keep_raw: bool = False,
    # Optional overrides for specific hadm_id and vital keys, e.g. {25921885: {"oxygen_saturation": "98"}}
    hadm_id_overrides: Optional[Dict[int, Dict[str, str]]] = None,
) -> None: 
    hadm_id_overrides = hadm_id_overrides or {}

    n_in = 0
    n_out = 0

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line_num, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSON on line {line_num}: {e}")
                continue

            n_in += 1

            subject_id = data.get("subject_id")
            hadm_id = data.get("hadm_id")

            vitals = data.get("vitals", {}) or {}

            # Apply hadm_id-specific overrides if provided
            try:
                hid_int = int(hadm_id) if hadm_id is not None else None
            except Exception:
                hid_int = None

            if hid_int is not None and hid_int in hadm_id_overrides:
                for vital_key, override_value in hadm_id_overrides[hid_int].items():
                    vitals[vital_key] = override_value

            cleaned_vitals: Dict[str, str] = {}
            for vital_name in VITAL_KEYS:
                cleaned_vitals[vital_name] = clean_vital(vital_name, vitals.get(vital_name, "NaN"))

            out_obj: Dict[str, Any] = {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "vitals": cleaned_vitals,
            }
            if keep_raw:
                out_obj["raw_vitals"] = vitals

            outfile.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Cleaned {n_out} entries (read {n_in}). Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Clean extracted vitals JSONL")
    parser.add_argument("--input_file", required=True, help="Input JSONL from vLLM vitals extractor")
    parser.add_argument("--output_file", required=True, help="Output JSONL with cleaned vitals")
    parser.add_argument("--keep_raw", action="store_true", help="Include raw_vitals in output")
    args = parser.parse_args()

    # If you ever need overrides, add them here:
    # hadm_id_overrides = {
    #   25921885: {"oxygen_saturation": "98"},
    # }
    hadm_id_overrides: Dict[int, Dict[str, str]] = {}

    clean_vitals_file(
        input_path=args.input_file,
        output_path=args.output_file,
        keep_raw=args.keep_raw,
        hadm_id_overrides=hadm_id_overrides,
    )


if __name__ == "__main__":
    main()