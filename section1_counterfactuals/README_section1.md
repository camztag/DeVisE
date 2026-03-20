# Section 1: Data Preparation and Counterfactual Generation

This section builds the ICU cohort, derives admission-style notes from discharge summaries, extracts and cleans vital signs, generates counterfactual dictionaries, creates note-level and template-level counterfactuals, and runs validation checks.

Recommended entry point:

- top-level [`run_section1.sh`](../run_section1.sh)

This README reflects the scripts currently present in this repository:

- `1-cohort-data-extraction.py`
- `2-clean-notes.py`
- `3-vitals-extraction.py`
- `4-clean-vitals.py`
- `5-counterfactual-dictionary.py`
- `6-create-counterfactuals.py`
- `7-counterfactuals-checks.py`
- `8-templates-generation-and-counterfactuals.py`

## 1. Data Access and Download

The data source is MIMIC-IV v3.1 and MIMIC-IV-Note v2.2.

- Structured tables: [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)
- Notes: [MIMIC-IV-Note v2.2](https://www.physionet.org/content/mimic-iv-note/2.2/)

You need PhysioNet access and credentialed access to download the files.

After download, place the required CSV files in a single local data directory. The scripts assume these filenames:

- `icustays.csv`
- `admissions.csv`
- `patients.csv`
- `discharge.csv`

`discharge.csv` is used as the source note file from which admission-style notes are derived during preprocessing.

## 2. Environment

Python 3 is required.

Install the main Python dependencies used in this section:

```bash
pip install -r requirements.txt
```

Set a shared data directory once for the whole pipeline:

```bash
export SECTION1_RAW_DATA_DIR="/absolute/path/to/data/raw"
export SECTION1_DATA_DIR="/absolute/path/to/data/processed/section1"
```

If you do not set them:

- `SECTION1_DATA_DIR` defaults to `./data/processed/section1`
- `SECTION1_RAW_DATA_DIR` defaults to the same location as `SECTION1_DATA_DIR`

Recommended layout:

```text
data/
├── raw/
│   ├── icustays.csv
│   ├── admissions.csv
│   ├── patients.csv
│   └── discharge.csv
└── processed/
    └── section1/
```

Recommended entry point for this layout:

```bash
bash ./run_section1.sh \
  --raw-data-dir ./data/raw \
  --output-dir ./data/processed/section1
```

## 3. Pipeline Overview

The scripted Section 1 pipeline is:

1. Build the ICU cohort table.
2. Extract and clean admission-style note sections.
3. Extract vital signs from the cleaned notes with an LLM.
4. Clean the extracted vital signs.
5. Generate counterfactual dictionaries from cleaned vital signs.
6. Create note-level counterfactuals by replacing matched spans in the original note text.
7. Run validation checks on the generated counterfactual notes.
8. Build original structured templates, vital-sign counterfactual templates, and demographic counterfactual templates.

The easiest way to run the full section is the top-level runner:

```bash
bash ./run_section1.sh \
  --raw-data-dir ./data/raw \
  --output-dir ./data/processed/section1
```

## 4. Run Order

### Step 1. Build the structured cohort

This script:

- merges ICU stays within the same `hadm_id` if gaps are less than 48 hours
- keeps only the first ICU episode per `hadm_id`
- excludes episodes shorter than 24 hours
- adds demographics and mortality during ICU stay
- restricts to admissions that have a discharge note

Run with the top-level runner or explicitly pass paths. If you call the script directly, use:

```bash
python3 'section1_counterfactuals/1-cohort-data-extraction.py' \
  --icustays_csv "$SECTION1_RAW_DATA_DIR/icustays.csv" \
  --admissions_csv "$SECTION1_RAW_DATA_DIR/admissions.csv" \
  --patients_csv "$SECTION1_RAW_DATA_DIR/patients.csv" \
  --discharge_csv "$SECTION1_RAW_DATA_DIR/discharge.csv" \
  --output_csv "$SECTION1_DATA_DIR/icu_cohort_data.csv"
```

Output:

- `icu_cohort_data.csv`

### Step 2. Extract and clean admission-note sections

This script:

- keeps only the admission-relevant sections
- removes discharge and future-event language
- removes mortality-related mentions
- keeps only notes with a non-empty `PHYSICAL EXAM` section after preprocessing

Run with the top-level runner or explicitly pass paths. If you call the script directly, use:

```bash
python3 'section1_counterfactuals/2-clean-notes.py' \
  --discharge_csv "$SECTION1_RAW_DATA_DIR/discharge.csv" \
  --cohort_csv "$SECTION1_DATA_DIR/icu_cohort_data.csv" \
  --raw_output "$SECTION1_DATA_DIR/icu_notes_admission_raw.jsonl" \
  --clean_output "$SECTION1_DATA_DIR/icu_notes_admission_clean.jsonl"
```

Outputs:

- `icu_notes_admission_raw.jsonl`
- `icu_notes_admission_clean.jsonl`

### Optional sampling before vital extraction

`run_section1.sh` supports sampling a subset of cleaned notes via `--sample-size`. This is useful for quick tests before running the full pipeline.

Example:

```bash
bash ./run_section1.sh \
  --raw-data-dir ./data/raw \
  --output-dir ./data/processed/section1 \
  --sample-size 1000
```
Sample-specific outputs follow the same naming pattern with the `_sampleN` suffix.

### Step 3. Extract vitals from notes with the LLM

This step requires access to an LLM backend and is typically the most compute-intensive part of Section 1.

This script reads the `PHYSICAL EXAM` section and extracts:

- `temperature`
- `heart_rate`
- `blood_pressure`
- `respiration_rate`
- `oxygen_saturation`

Example on the full cleaned notes:

```bash
python3 'section1_counterfactuals/3-vitals-extraction.py' \
  --input_file "$SECTION1_DATA_DIR/icu_notes_admission_clean.jsonl" \
  --output_file "$SECTION1_DATA_DIR/extracted_vitals.jsonl" \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --tp 2 \
  --batch_size 16
```
It is recommended to manually inspect a subset of extracted values before continuing with cleaning and counterfactual generation.

### Step 4. Clean extracted vitals

This script normalizes extracted vital signs into numeric or standardized forms.

Use `--keep_raw` so downstream work can preserve the original surface form.

Full set:

```bash
python3 'section1_counterfactuals/4-clean-vitals.py' \
  --input_file "$SECTION1_DATA_DIR/extracted_vitals.jsonl" \
  --output_file "$SECTION1_DATA_DIR/cleaned_vitals.jsonl" \
  --keep_raw
```

Main output fields include:
- `subject_id`
- `hadm_id`
- `vitals` with cleaned values
- `raw_vitals` if `--keep_raw` is used

### Step 5. Create counterfactual dictionaries

This script classifies cleaned vitals into bins and samples replacement values for each class.

Full set:

```bash
python3 'section1_counterfactuals/5-counterfactual-dictionary.py' \
  --input_file "$SECTION1_DATA_DIR/cleaned_vitals.jsonl" \
  --output_dir "$SECTION1_DATA_DIR/counterfactuals"
```

Outputs:

- `oxygen_saturation_counterfactuals.jsonl`
- `blood_pressure_counterfactuals.jsonl`
- `temperature_counterfactuals.jsonl`
- `respiration_rate_counterfactuals.jsonl`
- `heart_rate_counterfactuals.jsonl`

### Step 6. Create note-level counterfactuals

This script uses:

- the clean admission notes
- the vitals file
- the counterfactual dictionaries

and replaces the matched vital mention inside the `PHYSICAL EXAM` section.

Full set:

```bash
python3 'section1_counterfactuals/6-create-counterfactuals.py' \
  --notes_jsonl "$SECTION1_DATA_DIR/icu_notes_admission_clean.jsonl" \
  --vitals_jsonl "$SECTION1_DATA_DIR/cleaned_vitals.jsonl" \
  --cf_dir "$SECTION1_DATA_DIR/counterfactuals" \
  --output_dir "$SECTION1_DATA_DIR/counterfactuals"
```

Outputs:

- `cf_bp.jsonl`
- `cf_hr.jsonl`
- `cf_rr.jsonl`
- `cf_os.jsonl`
- `cf_tp.jsonl`
- `not_found_hadm_ids.txt`
- `none_matched_vitals.txt`
- `oxygen_saturation_not_matched.txt`

### Step 7. Validate note-level counterfactuals

This script merges the generated counterfactual note files and reports:

- rows with duplicate original vital values
- counterfactuals with zero, multiple, or out-of-section changes
- missing original notes

Full set:

```bash
python3 'section1_counterfactuals/7-counterfactuals-checks.py' \
  --original_notes_jsonl "$SECTION1_DATA_DIR/icu_notes_admission_clean.jsonl" \
  --original_vitals_jsonl "$SECTION1_DATA_DIR/cleaned_vitals.jsonl" \
  --cf_dir "$SECTION1_DATA_DIR/counterfactuals"
```

Main outputs:

- `cf_all.jsonl`
- `cf_all_diffs.jsonl`
- `duplicate_original_vitals.jsonl`
- `problematic_counterfactuals.jsonl`
- `cf_summary.json`
- `missing_original_notes.txt`

Note: manual inspection of reports is recommended.

### Step 8. Create template-based original and counterfactual data

This script combines cohort demographics with cleaned vitals and creates:

- original structured templates
- vital-sign counterfactual structured templates
- demographic counterfactual structured templates

This step depends on the cohort file, cleaned vitals, and the counterfactual dictionaries generated earlier in the pipeline.

Full set:

```bash
python3 'section1_counterfactuals/8-templates-generation-and-counterfactuals.py' \
  --cohort_csv "$SECTION1_DATA_DIR/icu_cohort_data.csv" \
  --cleaned_vitals_jsonl "$SECTION1_DATA_DIR/cleaned_vitals.jsonl" \
  --cf_dir "$SECTION1_DATA_DIR/counterfactuals"
```

Outputs:

- `original_notes_template_based.jsonl`
- `counterfactual_notes_template_based.jsonl`
- `demographics_counterfactual_notes_template_based.jsonl`

## 5. Data Products Summary

Core files produced by this section:

- `icu_cohort_data.csv`
- `icu_notes_admission_raw.jsonl`
- `icu_notes_admission_clean.jsonl`
- `extracted_vitals.jsonl`
- `cleaned_vitals.jsonl`
- `counterfactuals/*.jsonl`
- `counterfactuals/cf_*.jsonl`
- `counterfactuals/cf_summary.json`
- `original_notes_template_based.jsonl`
- `counterfactual_notes_template_based.jsonl`
- `demographics_counterfactual_notes_template_based.jsonl`

## 6. Reproducibility Notes

For reproducible runs:

- keep all raw input CSVs in one fixed directory
- set `SECTION1_DATA_DIR` explicitly
- save the exact model name used in step 3
- keep `raw_vitals` when cleaning vitals

Example:

```bash
python3 section1_counterfactuals/5-counterfactual-dictionary.py \
  --input_file "$SECTION1_DATA_DIR/cleaned_vitals.jsonl" \
  --output_dir "$SECTION1_DATA_DIR/counterfactuals" \
```

## 7. Notes

- Section 1 produces the data inputs used by Section 2.
- If you run the pipeline in sample mode, downstream Section 2 runs should use the corresponding `_sampleN` files.