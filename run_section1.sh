#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION1_DIR="$SCRIPT_DIR/section1_counterfactuals"

usage() {
  cat <<'EOF'
Usage:
  bash run_section1.sh [options]

Options:
  --raw-data-dir PATH        Directory containing raw source CSVs. Defaults to SECTION1_RAW_DATA_DIR or <output-dir>
  --output-dir PATH          Directory for Section 1 outputs. Defaults to SECTION1_DATA_DIR or ./data/processed/section1
  --notes-input PATH         Override notes input for step 3/6/7. Default: <data-dir>/icu_notes_admission_clean.jsonl
  --sample-size N            If set, randomly sample N notes after step 2
  --sample-seed N            Seed for sampling. Default: 42
  --model NAME               Model for 3-vitals-extraction.py
  --tp N                     Tensor parallel size for 3-vitals-extraction.py. Default: 2
  --batch-size N             Batch size for 3-vitals-extraction.py. Default: 16
  --max-model-len N          Max model length for 3-vitals-extraction.py. Default: 4896
  --cf-samples N             Counterfactual samples per class in step 5. Default: 5
  --cf-seed N                Seed for step 5. Default: 42
  --skip-cohort              Skip step 1
  --skip-notes               Skip step 2
  --skip-vitals-extract      Skip step 3
  --skip-vitals-clean        Skip step 4
  --skip-cf-dict             Skip step 5
  --skip-cf-notes            Skip step 6
  --skip-cf-checks           Skip step 7
  --skip-templates           Skip step 8
  --help                     Show this help

Examples:
  bash run_section1.sh --raw-data-dir /abs/path/to/raw --output-dir /abs/path/to/processed/section1
  bash run_section1.sh --raw-data-dir /abs/path/to/raw --output-dir /abs/path/to/processed/section1 --sample-size 1000
EOF
}

OUTPUT_DIR="${SECTION1_DATA_DIR:-$(pwd)/data/processed/section1}"
RAW_DATA_DIR="${SECTION1_RAW_DATA_DIR:-}"
SAMPLE_SIZE=""
SAMPLE_SEED="42"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
TP="2"
BATCH_SIZE="16"
MAX_MODEL_LEN="4896"
CF_SAMPLES="5"
CF_SEED="42"

SKIP_COHORT="0"
SKIP_NOTES="0"
SKIP_VITALS_EXTRACT="0"
SKIP_VITALS_CLEAN="0"
SKIP_CF_DICT="0"
SKIP_CF_NOTES="0"
SKIP_CF_CHECKS="0"
SKIP_TEMPLATES="0"

NOTES_INPUT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-data-dir)
      RAW_DATA_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --notes-input)
      NOTES_INPUT_OVERRIDE="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --sample-seed)
      SAMPLE_SEED="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --tp)
      TP="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --cf-samples)
      CF_SAMPLES="$2"
      shift 2
      ;;
    --cf-seed)
      CF_SEED="$2"
      shift 2
      ;;
    --skip-cohort)
      SKIP_COHORT="1"
      shift
      ;;
    --skip-notes)
      SKIP_NOTES="1"
      shift
      ;;
    --skip-vitals-extract)
      SKIP_VITALS_EXTRACT="1"
      shift
      ;;
    --skip-vitals-clean)
      SKIP_VITALS_CLEAN="1"
      shift
      ;;
    --skip-cf-dict)
      SKIP_CF_DICT="1"
      shift
      ;;
    --skip-cf-notes)
      SKIP_CF_NOTES="1"
      shift
      ;;
    --skip-cf-checks)
      SKIP_CF_CHECKS="1"
      shift
      ;;
    --skip-templates)
      SKIP_TEMPLATES="1"
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)/$(basename "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_DIR"
if [[ -z "$RAW_DATA_DIR" ]]; then
  RAW_DATA_DIR="$OUTPUT_DIR"
fi
RAW_DATA_DIR="$(cd "$(dirname "$RAW_DATA_DIR")" && pwd)/$(basename "$RAW_DATA_DIR")"
export SECTION1_DATA_DIR="$OUTPUT_DIR"
export SECTION1_RAW_DATA_DIR="$RAW_DATA_DIR"

RAW_NOTES="$OUTPUT_DIR/icu_notes_admission_raw.jsonl"
CLEAN_NOTES="$OUTPUT_DIR/icu_notes_admission_clean.jsonl"

if [[ -n "$NOTES_INPUT_OVERRIDE" ]]; then
  NOTES_INPUT="$NOTES_INPUT_OVERRIDE"
else
  NOTES_INPUT="$CLEAN_NOTES"
fi

if [[ -n "$SAMPLE_SIZE" ]]; then
  NOTES_INPUT="$OUTPUT_DIR/icu_notes_admission_clean_sample${SAMPLE_SIZE}.jsonl"
  EXTRACTED_VITALS="$OUTPUT_DIR/extracted_vitals_sample${SAMPLE_SIZE}.jsonl"
  CLEANED_VITALS="$OUTPUT_DIR/cleaned_vitals_sample${SAMPLE_SIZE}.jsonl"
  CF_DIR="$OUTPUT_DIR/counterfactuals_sample${SAMPLE_SIZE}"
  TEMPLATE_ORIG="$OUTPUT_DIR/original_notes_template_based_sample${SAMPLE_SIZE}.jsonl"
  TEMPLATE_CF="$OUTPUT_DIR/counterfactual_notes_template_based_sample${SAMPLE_SIZE}.jsonl"
  TEMPLATE_DEMOGRAPHICS_CF="$OUTPUT_DIR/demographics_counterfactual_notes_template_based_sample${SAMPLE_SIZE}.jsonl"
else
  EXTRACTED_VITALS="$OUTPUT_DIR/extracted_vitals.jsonl"
  CLEANED_VITALS="$OUTPUT_DIR/cleaned_vitals.jsonl"
  CF_DIR="$OUTPUT_DIR/counterfactuals"
  TEMPLATE_ORIG="$OUTPUT_DIR/original_notes_template_based.jsonl"
  TEMPLATE_CF="$OUTPUT_DIR/counterfactual_notes_template_based.jsonl"
  TEMPLATE_DEMOGRAPHICS_CF="$OUTPUT_DIR/demographics_counterfactual_notes_template_based.jsonl"
fi

mkdir -p "$CF_DIR"

run_step() {
  echo
  echo "[SECTION1] $1"
  shift
  "$@"
}

if [[ "$SKIP_COHORT" != "1" ]]; then
  run_step "Step 1: cohort extraction" \
    python3 "$SECTION1_DIR/1-cohort-data-extraction.py" \
      --icustays_csv "$RAW_DATA_DIR/icustays.csv" \
      --admissions_csv "$RAW_DATA_DIR/admissions.csv" \
      --patients_csv "$RAW_DATA_DIR/patients.csv" \
      --discharge_csv "$RAW_DATA_DIR/discharge.csv" \
      --output_csv "$OUTPUT_DIR/icu_cohort_data.csv"
fi

if [[ "$SKIP_NOTES" != "1" ]]; then
  run_step "Step 2: admission note cleaning" \
    python3 "$SECTION1_DIR/2-clean-notes.py" \
      --discharge_csv "$RAW_DATA_DIR/discharge.csv" \
      --cohort_csv "$OUTPUT_DIR/icu_cohort_data.csv" \
      --raw_output "$RAW_NOTES" \
      --clean_output "$CLEAN_NOTES"
fi

if [[ -n "$SAMPLE_SIZE" ]]; then
  run_step "Sampling ${SAMPLE_SIZE} cleaned notes" \
    python3 -c "import json, random, sys; random.seed(int(sys.argv[3])); rows=[line for line in open(sys.argv[1], 'r', encoding='utf-8') if line.strip()]; n=min(int(sys.argv[2]), len(rows)); sample=random.sample(rows, n); open(sys.argv[4], 'w', encoding='utf-8').writelines(sample)" \
      "$CLEAN_NOTES" "$SAMPLE_SIZE" "$SAMPLE_SEED" "$NOTES_INPUT"
fi

if [[ "$SKIP_VITALS_EXTRACT" != "1" ]]; then
  run_step "Step 3: vitals extraction" \
    python3 "$SECTION1_DIR/3-vitals-extraction.py" \
      --input_file "$NOTES_INPUT" \
      --output_file "$EXTRACTED_VITALS" \
      --model "$MODEL" \
      --tp "$TP" \
      --batch_size "$BATCH_SIZE" \
      --max_model_len "$MAX_MODEL_LEN"
fi

if [[ "$SKIP_VITALS_CLEAN" != "1" ]]; then
  run_step "Step 4: vital cleaning" \
    python3 "$SECTION1_DIR/4-clean-vitals.py" \
      --input_file "$EXTRACTED_VITALS" \
      --output_file "$CLEANED_VITALS" \
      --keep_raw
fi

if [[ "$SKIP_CF_DICT" != "1" ]]; then
  run_step "Step 5: counterfactual dictionaries" \
    python3 "$SECTION1_DIR/5-counterfactual-dictionary.py" \
      --input_file "$CLEANED_VITALS" \
      --output_dir "$CF_DIR" \
      --n_samples_per_class "$CF_SAMPLES" \
      --seed "$CF_SEED"
fi

if [[ "$SKIP_CF_NOTES" != "1" ]]; then
  run_step "Step 6: note-level counterfactuals" \
    python3 "$SECTION1_DIR/6-create-counterfactuals.py" \
      --notes_jsonl "$NOTES_INPUT" \
      --vitals_jsonl "$CLEANED_VITALS" \
      --cf_dir "$CF_DIR" \
      --output_dir "$CF_DIR"
fi

if [[ "$SKIP_CF_CHECKS" != "1" ]]; then
  run_step "Step 7: counterfactual checks" \
    python3 "$SECTION1_DIR/7-counterfactuals-checks.py" \
      --original_notes_jsonl "$NOTES_INPUT" \
      --original_vitals_jsonl "$CLEANED_VITALS" \
      --cf_dir "$CF_DIR"
fi

if [[ "$SKIP_TEMPLATES" != "1" ]]; then
  run_step "Step 8: template generation" \
    python3 "$SECTION1_DIR/8-templates-generation-and-counterfactuals.py" \
      --cohort_csv "$OUTPUT_DIR/icu_cohort_data.csv" \
      --cleaned_vitals_jsonl "$CLEANED_VITALS" \
      --cf_dir "$CF_DIR" \
      --original_output "$TEMPLATE_ORIG" \
      --counterfactual_output "$TEMPLATE_CF" \
      --demographics_counterfactual_output "$TEMPLATE_DEMOGRAPHICS_CF"
fi

echo
echo "[SECTION1] Pipeline completed"
echo "[SECTION1] Raw data directory: $RAW_DATA_DIR"
echo "[SECTION1] Output directory: $OUTPUT_DIR"
