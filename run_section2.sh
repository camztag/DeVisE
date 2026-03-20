#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash run_section2.sh [options]

Required:
  --input_root PATH          Root directory containing Section 1 outputs
  --output_root PATH         Directory where Section 2 outputs will be written
  --backend NAME             One of: vllm, openai_compat

Backend-specific:
  --model_path NAME        Required for --backend vllm
  --model NAME             Optional for --backend openai_compat, default: openai/gpt-oss-120b

Optional:
  --model_key KEY            Stable short model key used in output filenames and Section 3
  --sample-size N            Use Section 1 sample outputs with suffix _sampleN
  --model NAME               Model exposed by the OpenAI-compatible server (for --backend openai_compat)
  --base_url URL             Base URL for OpenAI-compatible server. Default: http://localhost:8000/v1
  --api_key KEY              API key for OpenAI-compatible server. Default: EMPTY
  --prompt_type TYPE         One of: llama, deepseek, phi, qwen. Default: llama
  --model_profile NAME       Model profile from section2_utils.py. Default: default_2gpu
  --batch_size N             Batch size. Default: 32
  --los_bins SPEC            Optional LOS bins override. If omitted, quartiles are derived from <input_root>/icu_cohort_data.csv
  --structured_format TYPE   For templates: json or text. Default: json
  --sleep_s N                Sleep between OpenAI-compatible requests. Default: 0
  --run_raw                  Run raw experiments only
  --run_template             Run template experiments only
  --skip_original            Skip original datasets
  --skip_counterfactual      Skip counterfactual datasets
  --skip_demographics_cf     Skip demographics counterfactual dataset
  --skip_task                Skip mortality and LOS experiments
  --skip_nontask             Skip task-independent experiments
  --help                     Show this help

Examples:
  bash run_section2.sh \
    --input_root /abs/path/to/section1_outputs \
    --output_root /abs/path/to/section2_outputs \
    --model_path meta-llama/Llama-3.3-70B-Instruct
EOF
}

INPUT_ROOT=""
OUTPUT_ROOT=""
BACKEND="vllm"
MODEL_PATH=""
MODEL_KEY=""
SAMPLE_SIZE=""
MODEL=""
BASE_URL="http://localhost:8000/v1"
API_KEY="EMPTY"
PROMPT_TYPE="llama"
MODEL_PROFILE="default_2gpu"
BATCH_SIZE="32"
LOS_BINS=""
STRUCTURED_FORMAT="json"
SLEEP_S="0"

RUN_RAW="0"
RUN_TEMPLATE="0"
SKIP_ORIGINAL="0"
SKIP_COUNTERFACTUAL="0"
SKIP_DEMOGRAPHICS_CF="0"
SKIP_TASK="0"
SKIP_NONTASK="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --model_key)
      MODEL_KEY="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --base_url)
      BASE_URL="$2"
      shift 2
      ;;
    --api_key)
      API_KEY="$2"
      shift 2
      ;;
    --prompt_type)
      PROMPT_TYPE="$2"
      shift 2
      ;;
    --model_profile)
      MODEL_PROFILE="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --los_bins)
      LOS_BINS="$2"
      shift 2
      ;;
    --structured_format)
      STRUCTURED_FORMAT="$2"
      shift 2
      ;;
    --sleep_s)
      SLEEP_S="$2"
      shift 2
      ;;
    --run_raw)
      RUN_RAW="1"
      shift
      ;;
    --run_template)
      RUN_TEMPLATE="1"
      shift
      ;;
    --skip_original)
      SKIP_ORIGINAL="1"
      shift
      ;;
    --skip_counterfactual)
      SKIP_COUNTERFACTUAL="1"
      shift
      ;;
    --skip_demographics_cf)
      SKIP_DEMOGRAPHICS_CF="1"
      shift
      ;;
    --skip_task)
      SKIP_TASK="1"
      shift
      ;;
    --skip_nontask)
      SKIP_NONTASK="1"
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

if [[ -z "$INPUT_ROOT" || -z "$OUTPUT_ROOT" ]]; then
  usage
  exit 1
fi

if [[ "$BACKEND" != "vllm" && "$BACKEND" != "openai_compat" ]]; then
  echo "Invalid --backend: $BACKEND" >&2
  exit 1
fi

if [[ "$BACKEND" == "vllm" && -z "$MODEL_PATH" ]]; then
  echo "--model_path is required for --backend vllm" >&2
  exit 1
fi

if [[ "$BACKEND" == "openai_compat" && -z "$MODEL" ]]; then
  MODEL="openai/gpt-oss-120b"
fi

if [[ "$RUN_RAW" == "0" && "$RUN_TEMPLATE" == "0" ]]; then
  RUN_RAW="1"
  RUN_TEMPLATE="1"
fi

INPUT_ROOT="$(cd "$(dirname "$INPUT_ROOT")" && pwd)/$(basename "$INPUT_ROOT")"
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

if [[ -n "$SAMPLE_SIZE" ]]; then
  RAW_ORIGINAL="$INPUT_ROOT/icu_notes_admission_clean_sample${SAMPLE_SIZE}.jsonl"
  RAW_COUNTERFACTUAL="$INPUT_ROOT/counterfactuals_sample${SAMPLE_SIZE}/cf_all.jsonl"
  TEMPLATE_ORIGINAL="$INPUT_ROOT/original_notes_template_based_sample${SAMPLE_SIZE}.jsonl"
  TEMPLATE_COUNTERFACTUAL="$INPUT_ROOT/counterfactual_notes_template_based_sample${SAMPLE_SIZE}.jsonl"
  TEMPLATE_DEMOGRAPHICS_CF="$INPUT_ROOT/demographics_counterfactual_notes_template_based_sample${SAMPLE_SIZE}.jsonl"
else
  RAW_ORIGINAL="$INPUT_ROOT/icu_notes_admission_clean.jsonl"
  RAW_COUNTERFACTUAL="$INPUT_ROOT/counterfactuals/cf_all.jsonl"
  TEMPLATE_ORIGINAL="$INPUT_ROOT/original_notes_template_based.jsonl"
  TEMPLATE_COUNTERFACTUAL="$INPUT_ROOT/counterfactual_notes_template_based.jsonl"
  TEMPLATE_DEMOGRAPHICS_CF="$INPUT_ROOT/demographics_counterfactual_notes_template_based.jsonl"
fi
COHORT_CSV="$INPUT_ROOT/icu_cohort_data.csv"

if [[ "$SKIP_TASK" != "1" && -z "$LOS_BINS" && ! -f "$COHORT_CSV" ]]; then
  echo "Missing $COHORT_CSV. Provide Section 1 cohort output or override with --los_bins." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

run_cmd() {
  echo
  echo "[SECTION2] $1"
  shift
  "$@"
}

run_task_experiments() {
  local data_path="$1"
  local modality="$2"
  local dataset_variant="$3"
  local include_demographics="$4"
  local extra_args=()
  local task_runner=""
  local nontask_runner=""
  local model_args=()

  if [[ "$include_demographics" == "1" ]]; then
    extra_args+=(--include_demographics)
  fi

  if [[ "$BACKEND" == "vllm" ]]; then
    task_runner="$SCRIPT_DIR/run_task_vllm.py"
    nontask_runner="$SCRIPT_DIR/run_nontask_vllm.py"
    model_args=(--model_path "$MODEL_PATH" --model_profile "$MODEL_PROFILE")
  else
    task_runner="$SCRIPT_DIR/run_task_openai_compat.py"
    nontask_runner="$SCRIPT_DIR/run_nontask_openai_compat.py"
    model_args=(--model "$MODEL" --base_url "$BASE_URL" --api_key "$API_KEY" --sleep_s "$SLEEP_S")
  fi

  if [[ -n "$MODEL_KEY" ]]; then
    model_args+=(--model_key "$MODEL_KEY")
  fi

  if [[ "$SKIP_TASK" != "1" ]]; then
    los_args=()
    if [[ -n "$LOS_BINS" ]]; then
      los_args+=(--los_bins "$LOS_BINS")
    elif [[ -f "$COHORT_CSV" ]]; then
      los_args+=(--cohort_csv "$COHORT_CSV")
    fi

    run_cmd "Task mortality | $modality | $dataset_variant | demographics=$include_demographics" \
      python3 "$task_runner" \
        --data "$data_path" \
        --task mortality \
        --modality "$modality" \
        --dataset_variant "$dataset_variant" \
        --prompt_type "$PROMPT_TYPE" \
        --batch_size "$BATCH_SIZE" \
        --structured_format "$STRUCTURED_FORMAT" \
        --output_dir "$OUTPUT_ROOT" \
        "${model_args[@]}" \
        "${extra_args[@]}"

    run_cmd "Task LOS | $modality | $dataset_variant | demographics=$include_demographics" \
      python3 "$task_runner" \
        --data "$data_path" \
        --task los \
        --modality "$modality" \
        --dataset_variant "$dataset_variant" \
        --prompt_type "$PROMPT_TYPE" \
        --batch_size "$BATCH_SIZE" \
        --structured_format "$STRUCTURED_FORMAT" \
        --output_dir "$OUTPUT_ROOT" \
        "${los_args[@]}" \
        "${model_args[@]}" \
        "${extra_args[@]}"
  fi

  if [[ "$SKIP_NONTASK" != "1" ]]; then
    run_cmd "Task-independent | $modality | $dataset_variant | demographics=$include_demographics" \
      python3 "$nontask_runner" \
        --data "$data_path" \
        --modality "$modality" \
        --dataset_variant "$dataset_variant" \
        --batch_size "$BATCH_SIZE" \
        --structured_format "$STRUCTURED_FORMAT" \
        --output_dir "$OUTPUT_ROOT" \
        "${model_args[@]}" \
        "${extra_args[@]}"
  fi
}

maybe_run_dataset() {
  local data_path="$1"
  local modality="$2"
  local dataset_variant="$3"
  local include_demographics="$4"

  if [[ ! -f "$data_path" ]]; then
    echo "[SECTION2] Skipping missing dataset: $data_path"
    return
  fi

  run_task_experiments "$data_path" "$modality" "$dataset_variant" "$include_demographics"
}

if [[ "$RUN_RAW" == "1" ]]; then
  if [[ "$SKIP_ORIGINAL" != "1" ]]; then
    maybe_run_dataset "$RAW_ORIGINAL" "raw" "original" "0"
  fi
  if [[ "$SKIP_COUNTERFACTUAL" != "1" ]]; then
    maybe_run_dataset "$RAW_COUNTERFACTUAL" "raw" "counterfactual" "0"
  fi
fi

if [[ "$RUN_TEMPLATE" == "1" ]]; then
  if [[ "$SKIP_ORIGINAL" != "1" ]]; then
    maybe_run_dataset "$TEMPLATE_ORIGINAL" "template" "original" "1"
  fi
  if [[ "$SKIP_COUNTERFACTUAL" != "1" ]]; then
    maybe_run_dataset "$TEMPLATE_COUNTERFACTUAL" "template" "counterfactual" "1"
  fi
  if [[ "$SKIP_DEMOGRAPHICS_CF" != "1" ]]; then
    maybe_run_dataset "$TEMPLATE_DEMOGRAPHICS_CF" "template" "demographics_cf" "1"
  fi
fi

echo
echo "[SECTION2] Finished"
echo "[SECTION2] Input root: $INPUT_ROOT"
echo "[SECTION2] Output root: $OUTPUT_ROOT"
