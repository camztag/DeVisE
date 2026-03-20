#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import json
import re

from section1_utils import DEFAULT_DATA_DIR, ensure_parent_dir


# =========================
# Admission-section extraction
# =========================
def extract_section(text: str, section_patterns: list[str]) -> str:
    """
    Extract one section from the original note text, from the matched section header
    until the next all-caps-like header.
    """
    if not isinstance(text, str):
        return ""

    lower_text = text.lower()
    start_idx = None
    matched_header = None

    for pattern in section_patterns:
        idx = lower_text.find(pattern.lower())
        if idx != -1 and (start_idx is None or idx < start_idx):
            start_idx = idx
            matched_header = pattern

    if start_idx is None:
        return ""

    search_start = start_idx + len(matched_header)

    # Find next header such as "\n\nHEADER:"
    next_header_match = re.search(
        r"\n\s*\n[^\n:]{1,100}:",
        text[search_start:],
        flags=re.IGNORECASE
    )

    if next_header_match:
        end_idx = search_start + next_header_match.start()
    else:
        end_idx = len(text)

    section_text = text[search_start:end_idx]

    section_text = (
        section_text
        .replace("\n", " ")
        .strip()
    )

    # Remove any trailing discharge-related content that may have been included in the section.
    section_text = re.sub(r"\bdischarge\b.*", "", section_text, flags=re.IGNORECASE)

    return section_text.strip()


def filter_admission_text(notes_df: pd.DataFrame) -> pd.DataFrame:
    notes_df = notes_df.copy()
    notes_df["text"] = notes_df["text"].astype(str)

    notes_df["chief_complaint"] = notes_df["text"].apply(
        lambda x: extract_section(x, ["chief complaint:"])
    )
    notes_df["present_illness"] = notes_df["text"].apply(
        lambda x: extract_section(x, ["present illness:"])
    )
    notes_df["medical_history"] = notes_df["text"].apply(
        lambda x: extract_section(x, ["medical history:"])
    )
    notes_df["medication_adm"] = notes_df["text"].apply(
        lambda x: extract_section(
            x,
            ["medications on admission:", "medication on admission:", "medication at admission:"]
        )
    )
    notes_df["allergies"] = notes_df["text"].apply(
        lambda x: extract_section(x, ["allergies:"])
    )
    notes_df["physical_exam"] = notes_df["text"].apply(
        lambda x: extract_section(x, ["physical exam:"])
    )
    notes_df["family_history"] = notes_df["text"].apply(
        lambda x: extract_section(x, ["family history:"])
    )
    notes_df["social_history"] = notes_df["text"].apply(
        lambda x: extract_section(x, ["social history:"])
    )

    # Only keep notes that have at least one of the main sections non-empty (chief complaint, present illness, medical history).
    notes_df = notes_df[
        (notes_df["chief_complaint"] != "")
        | (notes_df["present_illness"] != "")
        | (notes_df["medical_history"] != "")
    ].copy()

    notes_df["text"] = (
        "CHIEF COMPLAINT: " + notes_df["chief_complaint"]
        + "\n\nPRESENT ILLNESS: " + notes_df["present_illness"]
        + "\n\nMEDICAL HISTORY: " + notes_df["medical_history"]
        + "\n\nMEDICATION ON ADMISSION: " + notes_df["medication_adm"]
        + "\n\nALLERGIES: " + notes_df["allergies"]
        + "\n\nPHYSICAL EXAM: " + notes_df["physical_exam"]
        + "\n\nFAMILY HISTORY: " + notes_df["family_history"]
        + "\n\nSOCIAL HISTORY: " + notes_df["social_history"]
    )

    return notes_df


# =========================
# Cleaning - truncate sections at discharge-related keywords/patterns
# =========================
def truncate_section(text: str, section_patterns: list[str], next_sections: list[str], discharge_keywords: list[str]) -> str:
    lower_text = text.lower()
    start_idx = None
    matched_pattern = None

    for pattern in section_patterns:
        idx = lower_text.find(pattern.lower())
        if idx != -1 and (start_idx is None or idx < start_idx):
            start_idx = idx
            matched_pattern = pattern

    if start_idx is None:
        return text  # section not found

    # Find where this section ends
    end_idx = None
    for next_sec in next_sections:
        idx = lower_text.find(next_sec.lower(), start_idx + len(matched_pattern))
        if idx != -1 and (end_idx is None or idx < end_idx):
            end_idx = idx

    if end_idx is None:
        end_idx = len(text)

    section_text = text[start_idx:end_idx]
    section_lower = section_text.lower()

    # Regex for checkbox death pattern like [] 6: Dead
    checkbox_dead_pattern = re.compile(
        r'\[\s*?\](?:\s|\u00A0)*\d+\s*:\s*dead\b',
        re.IGNORECASE
    )

    cut_idx = None

    # Check keyword substrings
    for keyword in discharge_keywords:
        idx = section_lower.find(keyword)
        if idx != -1 and (cut_idx is None or idx < cut_idx):
            cut_idx = idx

    # Check regex pattern too
    match = checkbox_dead_pattern.search(section_text)
    if match:
        idx = match.start()
        if cut_idx is None or idx < cut_idx:
            cut_idx = idx

    # Cut section
    if cut_idx is not None:
        cleaned_section = section_text[:cut_idx].rstrip()
        after_section = text[end_idx:].lstrip()
        cleaned_text = text[:start_idx] + cleaned_section + "\n\n" + after_section
        return cleaned_text

    return text

# This applies the truncation to all main sections, using the same discharge-related keywords.
def clean_note_text(text: str) -> str:
    discharge_keywords = [
        'upon discharge', 'on discharge', 'discharge physical exam', 'discharge exam',
        'discharge pe', 'patient expired', 'patient died', ': expired', ': deceased',
        'at discharge', 'pt expired', 'patient deceased', 'pt died', 'pt deceased',
        '**deceased', 'n/a. expired', '= expired', 'patient was declared dead',
        'discharge (deceased)', 'expired', 'day of discharge - deceased', '[x] 6: dead',
        'dc as deceased', 'patient pronounced dead', 'pronounced dead', 'discharge:',
        'declared deceased', 'patient expired', '. expired', 'he died on', '. deceased',
        'deceased at', 'was pronounced dead', 'brain death', 'deseased exam',
        '. deceased.', 'deceased ___', 'time of death', '___ deceased',
        'died this admission', 'patient was confirmed to have expired',
        'patient pronounced deceased', 'decleared dead', 'decleared deceased',
        'and expired', 'wife was informed of his death', '(deceased)',
        'discharge summary', 'death note', 'expired on', 'death exam',
        'physical exam after death', 'discharge (death)', "patient's death",
        'pronouncing death:', 'deceased exam', 'documenting death',
        '(on morning of death)', 'death ___', 'day of death', 'declared dead',
        'prior to death', '= death', ': death', 'prior to her death',
        'declaration of death', 'deceased physical exam', 'pulseless, deceased',
        'discharge vs', 'discharge vital signs', 'discharge wgt:',
        'discharge physical:', 'discharge vitals', 'discharge physical exam',
        'discharge t ', 'discharge weight'
    ]

    text = truncate_section(
        text,
        section_patterns=['chief complaint'],
        next_sections=[
            'present illness', 'medical history', 'medication on admission',
            'medication at admission', 'allergies', 'physical exam',
            'family history', 'social history'
        ],
        discharge_keywords=discharge_keywords
    )

    text = truncate_section(
        text,
        section_patterns=['present illness'],
        next_sections=[
            'medical history', 'medication on admission', 'medication at admission',
            'allergies', 'physical exam', 'family history', 'social history'
        ],
        discharge_keywords=discharge_keywords
    )

    text = truncate_section(
        text,
        section_patterns=['medical history'],
        next_sections=[
            'medication on admission', 'medication at admission',
            'allergies', 'physical exam', 'family history', 'social history'
        ],
        discharge_keywords=discharge_keywords
    )

    text = truncate_section(
        text,
        section_patterns=['medication at admission', 'medication on admission'],
        next_sections=['allergies', 'physical exam', 'family history', 'social history'],
        discharge_keywords=discharge_keywords
    )

    text = truncate_section(
        text,
        section_patterns=['physical exam'],
        next_sections=['family history', 'social history'],
        discharge_keywords=discharge_keywords
    )

    return text


# =========================
# Keep-only-nonempty physical exam - check if the PHYSICAL EXAM section has any non-whitespace content after cleaning, and drop notes that don't.
# =========================
_PHYS_RE = re.compile(
    r"PHYSICAL EXAM:\s*(.*?)(?:\n\nFAMILY HISTORY:|\n\nSOCIAL HISTORY:|$)",
    flags=re.IGNORECASE | re.DOTALL
)

def has_nonempty_physical_exam(text: str) -> bool:
    m = _PHYS_RE.search(text or "")
    if not m:
        return False
    return len(m.group(1).strip()) > 0


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Extract and clean admission-note sections")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Base directory for input/output files")
    parser.add_argument("--discharge_csv", type=str, default=None, help="Optional override path for discharge.csv")
    parser.add_argument("--cohort_csv", type=str, default=None, help="Optional override path for icu_cohort_data.csv")
    parser.add_argument("--raw_output", type=str, default=None, help="Optional output path for raw admission notes JSONL")
    parser.add_argument("--clean_output", type=str, default=None, help="Optional output path for cleaned admission notes JSONL")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    discharge_csv = Path(args.discharge_csv).expanduser() if args.discharge_csv else data_dir / "discharge.csv"
    cohort_csv = Path(args.cohort_csv).expanduser() if args.cohort_csv else data_dir / "icu_cohort_data.csv"
    out_jsonl_raw = Path(args.raw_output).expanduser() if args.raw_output else data_dir / "icu_notes_admission_raw.jsonl"
    out_jsonl_clean = Path(args.clean_output).expanduser() if args.clean_output else data_dir / "icu_notes_admission_clean.jsonl"

    cohort = pd.read_csv(cohort_csv)
    cohort_pairs = cohort[["subject_id", "hadm_id"]].drop_duplicates()
    cohort_pairs["subject_id"] = cohort_pairs["subject_id"].astype(int)
    cohort_pairs["hadm_id"] = cohort_pairs["hadm_id"].astype(int)

    discharge = pd.read_csv(discharge_csv)

    discharge = discharge.dropna(subset=["subject_id", "hadm_id", "text"]).copy()
    discharge["subject_id"] = discharge["subject_id"].astype(int)
    discharge["hadm_id"] = discharge["hadm_id"].astype(int)

    # Only filter by cohort
    discharge = discharge.merge(
        cohort_pairs,
        on=["subject_id", "hadm_id"],
        how="inner"
    )

    discharge["text"] = discharge["text"].astype(str).str.strip()

    # 1. Extract admission sections
    notes_df = filter_admission_text(discharge)

    # 2. Write raw JSONL
    ensure_parent_dir(out_jsonl_raw)
    with open(out_jsonl_raw, "w", encoding="utf-8") as f:
        for _, row in notes_df.iterrows():
            obj = {
                "hadm_id": int(row["hadm_id"]),
                "subject_id": int(row["subject_id"]),
                "text": row["text"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 3. Clean + keep only notes with non-empty PHYSICAL EXAM
    kept = 0
    dropped = 0

    ensure_parent_dir(out_jsonl_clean)
    with open(out_jsonl_raw, "r", encoding="utf-8") as fin, open(out_jsonl_clean, "w", encoding="utf-8") as fout:
        for line in fin:
            note = json.loads(line)
            note["text"] = clean_note_text(note["text"])

            if has_nonempty_physical_exam(note["text"]):
                fout.write(json.dumps(note, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1

    print(f"Wrote raw JSONL: {out_jsonl_raw}")
    print(f"Wrote clean JSONL: {out_jsonl_clean}")
    print(f"Kept {kept} notes; dropped {dropped} without PHYSICAL EXAM.")


if __name__ == "__main__":
    main()
