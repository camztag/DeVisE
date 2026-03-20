#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash run_section3.sh [options]

Required:
  --results_dir PATH         Section 2 results directory
  --output_root PATH         Root directory for Section 3 outputs
  --models KEYS              Comma-separated model keys, e.g. llama,deepseek,gptoss120

Optional:
  --labels_path PATH        Section 1 cohort file, usually icu_cohort_data.csv
  --run_mortality           Run mortality analysis
  --run_los                 Run LOS analysis
  --run_nontask             Run task-independent analysis
  --run_demographics        Run demographics CF analysis
  --los_class_hours CSV     Override LOS expected class hours for LOS analysis
  --demo_class_hours CSV    Override LOS expected class hours for demographics analysis
  --skip_indiv              Do not save per-individual outputs
  --help                    Show this help

If no --run_* flag is provided, all analyses are run.
EOF
}

RESULTS_DIR=""
LABELS_PATH=""
OUTPUT_ROOT=""
MODELS=""
RUN_MORTALITY="0"
RUN_LOS="0"
RUN_NOTASK="0"
RUN_DEMOGRAPHICS="0"
LOS_CLASS_HOURS=""
DEMO_CLASS_HOURS=""
SAVE_INDIV="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results_dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --labels_path)
      LABELS_PATH="$2"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --run_mortality)
      RUN_MORTALITY="1"
      shift
      ;;
    --run_los)
      RUN_LOS="1"
      shift
      ;;
    --run_nontask)
      RUN_NOTASK="1"
      shift
      ;;
    --run_demographics)
      RUN_DEMOGRAPHICS="1"
      shift
      ;;
    --los_class_hours)
      LOS_CLASS_HOURS="$2"
      shift 2
      ;;
    --demo_class_hours)
      DEMO_CLASS_HOURS="$2"
      shift 2
      ;;
    --skip_indiv)
      SAVE_INDIV="0"
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

if [[ -z "$RESULTS_DIR" || -z "$OUTPUT_ROOT" || -z "$MODELS" ]]; then
  usage
  exit 1
fi

if [[ "$RUN_MORTALITY" == "0" && "$RUN_LOS" == "0" && "$RUN_NOTASK" == "0" && "$RUN_DEMOGRAPHICS" == "0" ]]; then
  RUN_MORTALITY="1"
  RUN_LOS="1"
  RUN_NOTASK="1"
  RUN_DEMOGRAPHICS="1"
fi

if [[ ("$RUN_MORTALITY" == "1" || "$RUN_LOS" == "1" || "$RUN_DEMOGRAPHICS" == "1") && -z "$LABELS_PATH" ]]; then
  echo "--labels_path is required for mortality, LOS, and demographics analyses." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

run_step() {
  echo
  echo "[SECTION3] $1"
  shift
  "$@"
}

COMMON_ARGS=(--results_dir "$RESULTS_DIR" --models "$MODELS")
if [[ -n "$LABELS_PATH" ]]; then
  COMMON_ARGS+=(--labels_path "$LABELS_PATH")
fi

if [[ "$RUN_MORTALITY" == "1" ]]; then
  MORT_DIR="$OUTPUT_ROOT/mortality"
  mkdir -p "$MORT_DIR"
  EXTRA_ARGS=()
  if [[ "$SAVE_INDIV" == "1" ]]; then
    EXTRA_ARGS+=(--save_indiv_vital --save_indiv_hadm)
  fi
  run_step "Mortality analysis" \
    python3 "$SCRIPT_DIR/full-analysis-mortality2.py" \
      "${COMMON_ARGS[@]}" \
      --output_dir "$MORT_DIR" \
      "${EXTRA_ARGS[@]}"
fi

if [[ "$RUN_LOS" == "1" ]]; then
  LOS_DIR="$OUTPUT_ROOT/los"
  mkdir -p "$LOS_DIR"
  EXTRA_ARGS=()
  if [[ -n "$LOS_CLASS_HOURS" ]]; then
    EXTRA_ARGS+=(--override_class_hours "$LOS_CLASS_HOURS")
  fi
  if [[ "$SAVE_INDIV" == "1" ]]; then
    EXTRA_ARGS+=(--save_indiv_vital --save_indiv_hadm)
  fi
  run_step "LOS analysis" \
    python3 "$SCRIPT_DIR/full-analysis-los3.py" \
      "${COMMON_ARGS[@]}" \
      --output_dir "$LOS_DIR" \
      "${EXTRA_ARGS[@]}"
fi

if [[ "$RUN_NOTASK" == "1" ]]; then
  NOTASK_DIR="$OUTPUT_ROOT/task_independent"
  mkdir -p "$NOTASK_DIR"
  run_step "Task-independent analysis" \
    python3 "$SCRIPT_DIR/full-analysis-notask.py" \
      --results_dir "$RESULTS_DIR" \
      --output_dir "$NOTASK_DIR" \
      --models "$MODELS"
fi

if [[ "$RUN_DEMOGRAPHICS" == "1" ]]; then
  DEMO_DIR="$OUTPUT_ROOT/demographics"
  mkdir -p "$DEMO_DIR"
  EXTRA_ARGS=()
  if [[ -n "$DEMO_CLASS_HOURS" ]]; then
    EXTRA_ARGS+=(--override_class_hours "$DEMO_CLASS_HOURS")
  fi
  run_step "Demographics CF analysis" \
    python3 "$SCRIPT_DIR/full-analysis-demo2.py" \
      "${COMMON_ARGS[@]}" \
      --output_dir "$DEMO_DIR" \
      "${EXTRA_ARGS[@]}"
fi

echo
echo "[SECTION3] Finished"
echo "[SECTION3] Results dir: $RESULTS_DIR"
echo "[SECTION3] Output root: $OUTPUT_ROOT"
