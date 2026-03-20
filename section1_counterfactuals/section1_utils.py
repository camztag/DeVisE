#!/usr/bin/env python3

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_DATA_DIR = Path(os.getenv("SECTION1_DATA_DIR", Path.cwd() / "data")).expanduser()

VITAL_KEYS = [
    "temperature",
    "heart_rate",
    "blood_pressure",
    "respiration_rate",
    "oxygen_saturation",
]

VITAL_TO_CODE = {
    "blood_pressure": "bl",
    "heart_rate": "he",
    "temperature": "te",
    "respiration_rate": "re",
    "oxygen_saturation": "ox",
}

CLASS_TO_NUM_BY_VITAL = {
    "bl": {
        "normal": 0,
        "low": -1,
        "very low/ltl": -2,
        "high": 1,
        "very high/lth": 2,
    },
    "he": {
        "normal": 0,
        "low": -1,
        "very low/ltl": -2,
        "high": 1,
        "very high/lth": 2,
    },
    "te": {
        "normal": 0,
        "low": -1,
        "very low/ltl": -2,
        "high": 1,
        "very high/lth": 2,
    },
    "re": {
        "normal": 0,
        "low": -1,
        "very low/ltl": -2,
        "high": 1,
        "very high/lth": 2,
    },
    "ox": {
        "normal": 0,
        "low": -1,
        "very low/ltl": -2,
    },
}

DEMOGRAPHIC_RACE_CLASSES = ["white", "black", "asian_pacific", "hispanic/latino", "other/unknown"]

DEMOGRAPHIC_AGE_CATEGORIES: Dict[str, Tuple[int, int]] = {
    "youngAdults": (18, 35),
    "middleAgedAdults": (36, 55),
    "olderAdults": (56, 75),
    "elderly": (76, 120),
}


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Bad JSON in {path} line {line_num}: {e}")
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_id_list(path: Path, ids: List[Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(str(x) for x in ids))


def normalize_class_name(class_name: Optional[str]) -> Optional[str]:
    if class_name is None:
        return None
    return str(class_name).strip().lower()


def get_class_severity(vital_type: str, class_name: Optional[str]) -> Optional[int]:
    if class_name is None:
        return None
    vital_code = VITAL_TO_CODE[vital_type]
    return CLASS_TO_NUM_BY_VITAL.get(vital_code, {}).get(normalize_class_name(class_name))


def demographic_age_category(age: Any) -> Optional[str]:
    try:
        age_int = int(float(age))
    except Exception:
        return None
    for category, (low, high) in DEMOGRAPHIC_AGE_CATEGORIES.items():
        if low <= age_int <= high:
            return category
    return None


def sample_demographic_ages(
    category: str,
    rng: random.Random,
    original_age: Optional[Any] = None,
    n: int = 5,
) -> List[int]:
    low, high = DEMOGRAPHIC_AGE_CATEGORIES[category]
    values = set()
    original_age_str = None if original_age is None else str(original_age)

    while len(values) < n:
        value = rng.randint(low, high)
        if original_age_str is not None and str(value) == original_age_str:
            continue
        values.add(value)

    return sorted(values)
