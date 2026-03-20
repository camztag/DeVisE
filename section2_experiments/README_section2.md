# Section 2: Experiments

This section runs the full experiment matrix over:

- unstructured admission notes (`raw`)
- structured template inputs (`template`)
- original inputs
- note/value counterfactual inputs
- demographic counterfactual structured inputs
- task-conditioned experiments for `mortality` and `los`
- task-independent scoring experiments

The recommended entry points for this section are:

- `run_task_vllm.py`
- `run_nontask_vllm.py`
- `run_task_openai_compat.py`
- `run_nontask_openai_compat.py`
- `run_section2.sh`

Each experiment produces a JSONL file containing model predictions (class labels and probabilities) for each input example. These outputs are used as input to Section 3 for behavioral analysis.

## 1. Inputs From Section 1

Section 2 expects the data products created in Section 1.

### Raw / unstructured inputs

- original notes: `icu_notes_admission_clean.jsonl`
- counterfactual notes: `counterfactuals/cf_all.jsonl` or the per-vital `cf_*.jsonl` files for per-vital runs

### Template / structured inputs

- original templates: `original_notes_template_based.jsonl`
- counterfactual templates: `counterfactual_notes_template_based.jsonl`
- demographic counterfactual templates: `demographics_counterfactual_notes_template_based.jsonl`

## 2. Experiment Dimensions

The experiment matrix is:

- `task`: `mortality`, `los`, `task_independent`
- `modality`: `raw`, `template`
- `dataset_variant`: `original`, `counterfactual`, `demographics_cf`
- `include_demographics`: only meaningful for `template`

Notes:

- `task_independent` computes likelihood-based metrics (e.g., perplexity) and uses `run_nontask_vllm.py`
- `mortality` and `los` use `run_task_vllm.py`
- for `template`, demographics (`age`, `sex`, `race`) are included in the input representation by default
- the `--include_demographics` flag controls prompt construction and output naming
- in the current experimental setup, template experiments are run with demographics enabled

## 3. LOS Buckets

LOS buckets are the quartile-based bins derived from the final cohort LOS in hours. These bins are used to convert continuous LOS values into classification targets for model prediction.

By default, the Section 2 runners derive them from `icu_cohort_data.csv` using the LOS hours in the final cohort.

The can be overridden explicitly:

```bash
--los_bins '24:38,39:59,60:112,113:657'
```

Format:

- comma-separated ranges
- each range is `LOW:HIGH`
- bucket indices are assigned in order as `1,2,3,4`

If you call the task runners directly for LOS experiments, pass either `--los_bins` or `--cohort_csv`. The recommended reproducible path is to pass the Section 1 cohort file or use `run_section2.sh`, which picks it up automatically from the Section 1 output directory.

## 4. Environments

This section now supports two backends:

- `vllm` backend requires GPU resources
- `openai_compat` backend requires a running server exposing an OpenAI-compatible API

### `vllm` backend

Use this for the standard local vLLM models.

Scripts:

- `run_task_vllm.py`
- `run_nontask_vllm.py`

### `openai_compat` backend

Use this for GPT-OSS or any model served behind an OpenAI-compatible `/v1/completions` endpoint.

Scripts:

- `run_task_openai_compat.py`
- `run_nontask_openai_compat.py`

This separation is recommended when the GPT-OSS/server-based workflow requires a different vLLM version or serving stack.

For task-conditioned GPT-OSS runs, the OpenAI-compatible task runner now uses the tokenizer-specific GPT-OSS chat template again via `transformers.AutoTokenizer(..., trust_remote_code=True)`.

Recommended setup:

- one environment for standard local vLLM experiments
- one separate environment for GPT-OSS / OpenAI-compatible server experiments

The shell wrapper supports both through `--backend`.

Install the standard vLLM environment with:

```bash
pip install -r requirements.txt
```

Install the separate GPT/OpenAI-compatible environment with:

```bash
pip install -r requirements-gpt.txt
```

## 5. Model Profiles

[`section2_utils.py`](section2_utils.py)

Available defaults:

- `default_1gpu`
- `default_2gpu`
- `gptoss_1gpu`
- `gptoss_2gpu`

These profiles define:

- `tensor_parallel_size`
- `dtype`
- `max_model_len`
- `enforce_eager`
- `trust_remote_code`

## 6. Recommended Outputs

Output filenames are auto-generated from:

- model name
- experiment type
- modality
- dataset variant
- demographics setting

Example pattern:

```text
<model>__<experiment_type>__<modality>__<dataset_variant>__<with|no>_demographics.jsonl
```

This is handled automatically if you provide `--output_dir` instead of `--output`.

Section 2 produces model outputs in JSONL format, where each row corresponds to a single input note or counterfactual.

Depending on the experiment type, outputs include:

- for task-conditioned experiments (`mortality`, `los`):
  - predicted class labels
  - class probabilities (e.g., `prob_1`–`prob_4`)

- for task-independent experiments:
  - token-level log-probabilities
  - perplexity

All outputs preserve metadata (e.g., `hadm_id`, `subject_id`, `id`) to enable downstream analysis in Section 3.

## 7. Run The Full Matrix

Use the shell wrapper:

```bash
bash 'section2_experiments/run_section2.sh' \
  --input_root /absolute/path/to/section1_outputs \
  --output_root /absolute/path/to/section2_outputs \
  --backend vllm \
  --model_path 'meta-llama/Llama-3.3-70B-Instruct' \
  --model_key llama \
  --prompt_type llama \
  --model_profile default_2gpu
```

OpenAI-compatible / GPT-OSS example:

```bash
bash 'section2_experiments/run_section2.sh' \
  --input_root /absolute/path/to/section1_outputs \
  --output_root /absolute/path/to/section2_outputs_gptoss \
  --backend openai_compat \
  --model 'openai/gpt-oss-120b' \
  --model_key gptoss120 \
  --base_url 'http://localhost:8000/v1' \
  --api_key 'EMPTY' \
  --prompt_type llama
```

If Section 1 was run in sample mode, add `--sample-size N` so the wrapper reads the `_sampleN` files produced by `run_section1.sh`.

Example:

```bash
bash 'section2_experiments/run_section2.sh' \
  --input_root /absolute/path/to/section1_outputs \
  --output_root /absolute/path/to/section2_outputs \
  --backend vllm \
  --model_path 'meta-llama/Llama-3.3-70B-Instruct' \
  --model_key llama \
  --prompt_type llama \
  --model_profile default_2gpu \
  --sample-size 1000
```

This will run:

- raw original: mortality, los, task-independent
- raw counterfactual: mortality, los, task-independent
- template original: mortality, los, task-independent
- template counterfactual: mortality, los, task-independent
- template demographic counterfactuals: mortality, los, task-independent if the file exists

## 8. Run Individual Experiments

### Mortality on raw original notes

```bash
python3 'section2_experiments/run_task_vllm.py' \
  --data '/path/to/icu_notes_admission_clean.jsonl' \
  --task mortality \
  --modality raw \
  --dataset_variant original \
  --model_path 'meta-llama/Llama-3.3-70B-Instruct' \
  --model_key llama \
  --prompt_type llama \
  --model_profile default_2gpu \
  --output_dir '/path/to/section2_outputs'
```

### Mortality on raw original notes with OpenAI-compatible / GPT-OSS backend

```bash
python3 'section2_experiments/run_task_openai_compat.py' \
  --data '/path/to/icu_notes_admission_clean.jsonl' \
  --task mortality \
  --modality raw \
  --dataset_variant original \
  --model 'openai/gpt-oss-120b' \
  --model_key gptoss120 \
  --tokenizer_name 'openai/gpt-oss-120b' \
  --base_url 'http://localhost:8000/v1' \
  --api_key 'EMPTY' \
  --output_dir '/path/to/section2_outputs'
```

### LOS on raw counterfactual notes

```bash
python3 'section2_experiments/run_task_vllm.py' \
  --data '/path/to/counterfactuals/cf_all.jsonl' \
  --task los \
  --modality raw \
  --dataset_variant counterfactual \
  --model_path 'meta-llama/Llama-3.3-70B-Instruct' \
  --model_key llama \
  --prompt_type llama \
  --model_profile default_2gpu \
  --cohort_csv '/path/to/icu_cohort_data.csv' \
  --output_dir '/path/to/section2_outputs'
```

### Task-independent scoring on original templates with demographics

```bash
python3 'section2_experiments/run_nontask_vllm.py' \
  --data '/path/to/original_notes_template_based.jsonl' \
  --modality template \
  --dataset_variant original \
  --include_demographics \
  --structured_format text \
  --model_path 'meta-llama/Llama-3.3-70B-Instruct' \
  --model_key llama \
  --model_profile default_2gpu \
  --output_dir '/path/to/section2_outputs'
```

### Task-independent scoring on original templates with GPT-OSS backend

```bash
python3 'section2_experiments/run_nontask_openai_compat.py' \
  --data '/path/to/original_notes_template_based.jsonl' \
  --modality template \
  --dataset_variant original \
  --include_demographics \
  --structured_format text \
  --model 'openai/gpt-oss-120b' \
  --model_key gptoss120 \
  --base_url 'http://localhost:8000/v1' \
  --api_key 'EMPTY' \
  --output_dir '/path/to/section2_outputs'
```

### Mortality on template counterfactuals without demographics

```bash
python3 'section2_experiments/run_task_vllm.py' \
  --data '/path/to/counterfactual_notes_template_based.jsonl' \
  --task mortality \
  --modality template \
  --dataset_variant counterfactual \
  --structured_format json \
  --model_path 'meta-llama/Llama-3.3-70B-Instruct' \
  --model_key llama \
  --prompt_type llama \
  --model_profile default_2gpu \
  --output_dir '/path/to/section2_outputs'
```

### LOS on template demographic counterfactuals with demographics

```bash
python3 'section2_experiments/run_task_vllm.py' \
  --data '/path/to/demographics_counterfactual_notes_template_based.jsonl' \
  --task los \
  --modality template \
  --dataset_variant demographics_cf \
  --include_demographics \
  --structured_format json \
  --model_path 'meta-llama/Llama-3.3-70B-Instruct' \
  --model_key llama \
  --prompt_type llama \
  --model_profile default_2gpu \
  --cohort_csv '/path/to/icu_cohort_data.csv' \
  --output_dir '/path/to/section2_outputs'
```

## 9. Metadata Preservation

All new runners preserve input metadata when present, including:

- `hadm_id`
- `subject_id`
- `id`
- counterfactual severity/class metadata from Section 1

This is important for linking predictions back to the exact original or counterfactual record.

## 10. Reproducibility Notes

For reproducible runs:

- keep the Section 1 cohort file alongside the Section 2 runs so LOS quartiles can be derived automatically
- record the exact model name and prompt type
- record the backend (`vllm` or `openai_compat`)
- use a stable `--model_key` such as `llama`, `deepseek`, or `gptoss120` so Section 3 can resolve files consistently
- keep original and counterfactual datasets separate via `dataset_variant`

## Models Used in the Paper

The experiments reported in the paper include:

- LLaMA 3.3 (70B)
- GPT-OSS (120B)
- DeepSeek-R1
- Qwen 2.5
- Meditron
- OpenBioLLM
- Phi-4

Exact configurations are defined via `model_key` and `model_profile`.

The `model_key` is used to standardize filenames and ensure compatibility with Section 3, where results are aggregated across models. Consistent naming is required for reproducible evaluation.