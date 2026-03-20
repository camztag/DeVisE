#!/usr/bin/env python3
"""
Build ICU cohort table from:
- icustays.csv
- admissions.csv
- patients.csv
- discharge.csv  (used ONLY to restrict cohort to (subject_id, hadm_id) pairs with a discharge note)

Outputs one row per (subject_id, hadm_id) ICU episode after applying cohort rules.

Cohort decisions:
1) For patients with multiple ICU stays associated with the same hadm_id:
   - stays occurring <48 hours apart are merged into a single episode
     (episode outtime extended; episode LOS computed from intime/outtime)
   - stays separated by >=48 hours are treated as distinct admissions, but only
     the first ICU episode is retained (later episodes dropped)
2) Exclude ICU episodes shorter than 24 hours
3) Keep only (subject_id, hadm_id) pairs that appear in discharge.csv

Adds demographics:
- gender, anchor_age (age), dod (from patients.csv)
- race (from admissions.csv)
- race_group via mapping
- age_group via bins

Mortality:
- mortality = 1 only if dod occurs during ICU episode (intime < dod <= outtime)
  else 0 (including dod missing)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from section1_utils import DEFAULT_DATA_DIR, ensure_parent_dir


# =========================
MERGE_GAP_HOURS = 48
MIN_EPISODE_HOURS = 24


# =========================
# RACE + AGE CATEGORIZATION
# =========================
def get_race_group(race: str) -> str:
    race_mapping = {
        "WHITE": "white",
        "WHITE - OTHER EUROPEAN": "white",
        "WHITE - RUSSIAN": "white",
        "WHITE - EASTERN EUROPEAN": "white",
        "WHITE - BRAZILIAN": "white",
        "BLACK/AFRICAN AMERICAN": "black",
        "BLACK/CAPE VERDEAN": "black",
        "BLACK/CARIBBEAN ISLAND": "black",
        "BLACK/AFRICAN": "black",
        "ASIAN": "asian_pacific",
        "ASIAN - CHINESE": "asian_pacific",
        "ASIAN - SOUTH EAST ASIAN": "asian_pacific",
        "ASIAN - ASIAN INDIAN": "asian_pacific",
        "ASIAN - KOREAN": "asian_pacific",
        "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "asian_pacific",
        "HISPANIC/LATINO - PUERTO RICAN": "hispanic/latino",
        "HISPANIC OR LATINO": "hispanic/latino",
        "HISPANIC/LATINO - DOMINICAN": "hispanic/latino",
        "HISPANIC/LATINO - GUATEMALAN": "hispanic/latino",
        "HISPANIC/LATINO - SALVADORAN": "hispanic/latino",
        "HISPANIC/LATINO - COLUMBIAN": "hispanic/latino",
        "HISPANIC/LATINO - MEXICAN": "hispanic/latino",
        "HISPANIC/LATINO - HONDURAN": "hispanic/latino",
        "HISPANIC/LATINO - CUBAN": "hispanic/latino",
        "HISPANIC/LATINO - CENTRAL AMERICAN": "hispanic/latino",
        "SOUTH AMERICAN": "hispanic/latino",
        "PORTUGUESE": "other/unknown",
        "MULTIPLE RACE/ETHNICITY": "other/unknown",
        "AMERICAN INDIAN/ALASKA NATIVE": "other/unknown",
        "UNKNOWN": "other/unknown",
        "UNABLE TO OBTAIN": "other/unknown",
        "PATIENT DECLINED TO ANSWER": "other/unknown",
        "OTHER": "other/unknown",
    }
    if pd.isna(race) or str(race).strip() == "":
        return "other/unknown"
    return race_mapping.get(str(race).upper(), "other/unknown")


age_categories = {
    "young adults": {"range": (18, 35)},
    "middle-aged adults": {"range": (36, 55)},
    "older adults": {"range": (56, 75)},
    "elderly": {"range": (76, 91)},
}

def get_age_group(age) -> str:
    if pd.isna(age):
        return "unknown"
    try:
        age = float(age)
    except Exception:
        return "unknown"
    for label, spec in age_categories.items():
        lo, hi = spec["range"]
        if lo <= age <= hi:
            return label
    if age < 18:
        return "<18"
    return ">91"


# =========================
# EPISODE MERGING
# =========================
def hours_between(a: pd.Timestamp, b: pd.Timestamp) -> float:
    return float((b - a) / pd.Timedelta(hours=1))

def merge_icustays_to_first_episode(stays: pd.DataFrame) -> pd.DataFrame:
    """
    Input: icustays rows for a single (subject_id, hadm_id)
    Output: 1-row dataframe for the FIRST ICU episode for that hadm_id, where
            consecutive stays with gap <48h are merged.
            If a later stay is >=48h apart, it is dropped (keep first episode only).
    """
    if stays.empty:
        return stays

    stays = stays.sort_values("intime").reset_index(drop=True)

    cur = stays.loc[0].to_dict()
    merged_stay_ids = [str(cur.get("stay_id", ""))]

    for i in range(1, len(stays)):
        nxt = stays.loc[i]
        gap_h = hours_between(cur["outtime"], nxt["intime"])
        if gap_h < MERGE_GAP_HOURS:
            cur["outtime"] = max(cur["outtime"], nxt["outtime"])
            merged_stay_ids.append(str(nxt.get("stay_id", "")))
        else:
            break

    episode_hours = hours_between(cur["intime"], cur["outtime"])
    cur["episode_los_hours"] = episode_hours
    cur["episode_los_days"] = episode_hours / 24.0
    cur["merged_stay_ids"] = "|".join([x for x in merged_stay_ids if x])

    return pd.DataFrame([cur])


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(description="Build ICU cohort data for Section 1")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Directory containing raw cohort CSVs")
    parser.add_argument("--icustays_csv", type=str, default=None, help="Optional override path for icustays.csv")
    parser.add_argument("--admissions_csv", type=str, default=None, help="Optional override path for admissions.csv")
    parser.add_argument("--patients_csv", type=str, default=None, help="Optional override path for patients.csv")
    parser.add_argument("--discharge_csv", type=str, default=None, help="Optional override path for discharge.csv")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional output path for ICU cohort CSV")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    icustays_csv = Path(args.icustays_csv).expanduser() if args.icustays_csv else data_dir / "icustays.csv"
    admissions_csv = Path(args.admissions_csv).expanduser() if args.admissions_csv else data_dir / "admissions.csv"
    patients_csv = Path(args.patients_csv).expanduser() if args.patients_csv else data_dir / "patients.csv"
    discharge_csv = Path(args.discharge_csv).expanduser() if args.discharge_csv else data_dir / "discharge.csv"
    out_csv = Path(args.output_csv).expanduser() if args.output_csv else data_dir / "icu_cohort_data.csv"

    # ---- Load discharge keys to restrict cohort ----
    disch_header = pd.read_csv(discharge_csv, nrows=0).columns.tolist()
    required = {"subject_id", "hadm_id"}
    missing = required - set(disch_header)
    if missing:
        raise ValueError(
            f"discharge.csv must contain columns {sorted(required)}. Missing: {sorted(missing)}"
        )

    df_discharge_keys = pd.read_csv(discharge_csv, usecols=["subject_id", "hadm_id"], low_memory=False).dropna()
    df_discharge_keys["subject_id"] = df_discharge_keys["subject_id"].astype(int)
    df_discharge_keys["hadm_id"] = df_discharge_keys["hadm_id"].astype(int)
    df_discharge_keys = df_discharge_keys.drop_duplicates(["subject_id", "hadm_id"])

    # ---- Load ICU stays ----
    icu = pd.read_csv(
        icustays_csv,
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
        low_memory=False,
    )
    icu["subject_id"] = icu["subject_id"].astype(int)
    icu["hadm_id"] = icu["hadm_id"].astype(int)

    # Restrict ICU stays to those with discharge notes (subject_id, hadm_id pairs)
    icu = icu.merge(df_discharge_keys, on=["subject_id", "hadm_id"], how="inner")

    # ---- Load admissions (race/ethnicity field may differ by version) ----
    adm_header = pd.read_csv(admissions_csv, nrows=0).columns.tolist()
    if "race" in adm_header:
        adm = pd.read_csv(admissions_csv, usecols=["subject_id", "hadm_id", "race"], low_memory=False)
        race_src = "race"
    elif "ethnicity" in adm_header:
        adm = pd.read_csv(admissions_csv, usecols=["subject_id", "hadm_id", "ethnicity"], low_memory=False)
        adm = adm.rename(columns={"ethnicity": "race"})
        race_src = "race"
    else:
        adm = pd.read_csv(admissions_csv, usecols=["subject_id", "hadm_id"], low_memory=False)
        adm["race"] = np.nan
        race_src = "race"

    adm["subject_id"] = adm["subject_id"].astype(int)
    adm["hadm_id"] = adm["hadm_id"].astype(int)

    # Restrict admissions to cohort keys too (optional but keeps things consistent)
    adm = adm.merge(df_discharge_keys, on=["subject_id", "hadm_id"], how="inner")

    # ---- Load patients ----
    pat = pd.read_csv(
        patients_csv,
        usecols=["subject_id", "gender", "anchor_age", "dod"],
        parse_dates=["dod"],
        low_memory=False,
    )
    pat["subject_id"] = pat["subject_id"].astype(int)

    # ---- Build first-episode-per-(subject_id, hadm_id) with within-hadm merging ----
    episodes = []
    for (sid, hid), grp in icu.groupby(["subject_id", "hadm_id"], sort=False):
        ep = merge_icustays_to_first_episode(grp)
        if not ep.empty:
            episodes.append(ep)

    if not episodes:
        out = pd.DataFrame(columns=[
            "subject_id","hadm_id","stay_id","merged_stay_ids","intime","outtime",
            "episode_los_days","episode_los_hours","gender","age","age_group",
            "race","race_group","dod","mortality",
        ])
        ensure_parent_dir(out_csv)
        out.to_csv(out_csv, index=False)
        print(f"Wrote: {out_csv} (0 rows)")
        return

    episodes = pd.concat(episodes, ignore_index=True)

    # ---- Exclude short episodes (<24h) ----
    episodes = episodes[episodes["episode_los_hours"] >= MIN_EPISODE_HOURS].copy()

    # ---- Join demographics ----
    episodes = episodes.merge(pat, on="subject_id", how="left")

    adm_one = (
        adm[["hadm_id", race_src]]
        .drop_duplicates("hadm_id")
        .rename(columns={race_src: "race"})
    )
    episodes = episodes.merge(adm_one, on="hadm_id", how="left")

    # ---- Categorize race + age ----
    episodes["age"] = episodes["anchor_age"]
    episodes["age_group"] = episodes["age"].apply(get_age_group)
    episodes["race_group"] = episodes["race"].apply(get_race_group)

    # ---- Mortality during ICU stay ----
    episodes["mortality"] = np.where(
        episodes["dod"].notna()
        & (episodes["dod"] >= episodes["intime"])
        & (episodes["dod"] <= episodes["outtime"]),
        1,
        0,
    )

    out = episodes[[
        "subject_id",
        "hadm_id",
        "stay_id",
        "merged_stay_ids",
        "intime",
        "outtime",
        "episode_los_days",
        "episode_los_hours",
        "gender",
        "age",
        "age_group",
        "race",
        "race_group",
        "dod",
        "mortality",
    ]].copy()

    out = out.sort_values(["subject_id", "hadm_id", "intime"]).reset_index(drop=True)
    ensure_parent_dir(out_csv)
    out.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
