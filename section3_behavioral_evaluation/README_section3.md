# Section 3: Behavioral Evaluation

This section analyzes Section 2 experiment outputs and produces behavioral evaluation summaries and plots for the DeVisE framework.

It evaluates whether model predictions change in a clinically meaningful and consistent way under controlled counterfactual perturbations.

The analysis focuses on:

- prediction shifts (e.g., mortality and LOS)
- likelihood shifts (e.g., perplexity, log-probabilities)
- directional consistency with clinical severity changes
- robustness and monotonicity across perturbations

The main supported analyses are:

- mortality robustness analysis (prediction stability and directional correctness)
- LOS robustness analysis (ΔE[LOS], KL divergence, monotonicity)
- task-independent likelihood analysis (perplexity and log-probability shifts)
- demographic counterfactual analysis (sensitivity to demographic changes)

## Entry Points and Scripts

Recommended entry point:

- [`run_section3.sh`](run_section3.sh)

Main analysis scripts:

- [`full-analysis-mortality2.py`](full-analysis-mortality2.py)
- [`full-analysis-los3.py`](full-analysis-los3.py)
- [`full-analysis-notask.py`](full-analysis-notask.py)
- [`full-analysis-demo2.py`](full-analysis-demo2.py)

## Behavioral Metrics

Section 3 computes several metrics to evaluate model behavior under counterfactual perturbations:

- Flip rate: percentage of predictions that change between original and counterfactual inputs
- Directional correctness: whether prediction shifts align with the direction of severity change
- KL divergence: difference between predicted probability distributions
- ΔE[LOS]: change in expected LOS (in hours) based on predicted class probabilities
- Monotonicity: whether model outputs change consistently with increasing severity


## 1. Inputs

Section 3 expects the outputs of Section 2 plus the cohort labels produced in Section 1.

### Results directory

`--results_dir` should point to a single Section 2 results directory, for example:

- `experiment_results/section2/vllm`
- `experiment_results/section2/openai_compat`

The analysis scripts now default to the new Section 2 filename pattern:

- mortality raw original: `{model}__mortality__raw__original__no_demographics.jsonl`
- mortality raw counterfactual: `{model}__mortality__raw__counterfactual__no_demographics.jsonl`
- LOS raw original: `{model}__los__raw__original__no_demographics.jsonl`
- LOS raw counterfactual: `{model}__los__raw__counterfactual__no_demographics.jsonl`
- task-independent template original: `{model}__task_independent__template__original__with_demographics.jsonl`
- task-independent template counterfactual: `{model}__task_independent__template__counterfactual__with_demographics.jsonl`
- demographics CF template: `{model}__los__template__demographics_cf__with_demographics.jsonl`

This is why Section 2 should be run with a stable `--model_key`.

### Labels file

- `data/processed/section1/icu_cohort_data.csv`
- the Section 1 cohort CSV, where labels come from `mortality` and `episode_los_hours`

Section 3 uses:

- `subject_id`
- `hadm_id`
- `mortality_label`
- `los_icu_hours`

## Outputs

Each analysis produces:

- summary tables (`.csv`) with aggregated metrics per model
- per-individual outputs for detailed inspection
- plots (`.png`, `.pdf`) showing behavior across severity levels

Typical outputs include:

- `summary_models.csv`
- `category_overview.csv`
- `plot_kl_severity_per_model.png`
- `plot_dE_hours_severity_per_model.png`
- `plot_overall_kl_per_model.png`

Outputs are organized by analysis type:

- `mortality/`
- `los/`
- `task_independent/`
- `demographics/`

## 2. Run All Analyses

```bash
bash 'section3_behavioral_evaluation/run_section3.sh' \
  --results_dir /absolute/path/to/experiment_results/section2/vllm \
  --labels_path /absolute/path/to/data/processed/section1/icu_cohort_data.csv \
  --output_root /absolute/path/to/experiment_results/section3 \
  --models llama,deepseek,gptoss120
```

If no `--run_*` flag is given, the script runs:

- mortality
- LOS
- task-independent
- demographics

`--labels_path` is required for mortality, LOS, and demographics analyses. It is not required when running only `--run_nontask`.

Outputs are written into subdirectories:

- `mortality/`
- `los/`
- `task_independent/`
- `demographics/`

## 3. Run Individual Analyses

### Mortality only

```bash
bash 'section3_behavioral_evaluation/run_section3.sh' \
  --results_dir /absolute/path/to/experiment_results/section2/vllm \
  --labels_path /absolute/path/to/data/processed/section1/icu_cohort_data.csv \
  --output_root /absolute/path/to/experiment_results/section3 \
  --models llama \
  --run_mortality
```

### LOS only

```bash
bash 'section3_behavioral_evaluation/run_section3.sh' \
  --results_dir /absolute/path/to/experiment_results/section2/vllm \
  --labels_path /absolute/path/to/data/processed/section1/icu_cohort_data.csv \
  --output_root /absolute/path/to/experiment_results/section3 \
  --models llama \
  --run_los
```

### Task-independent only

```bash
bash 'section3_behavioral_evaluation/run_section3.sh' \
  --results_dir /absolute/path/to/experiment_results/section2/vllm \
  --labels_path /absolute/path/to/data/processed/section1/icu_cohort_data.csv \
  --output_root /absolute/path/to/experiment_results/section3 \
  --models llama \
  --run_nontask
```

### Demographics only

```bash
bash 'section3_behavioral_evaluation/run_section3.sh' \
  --results_dir /absolute/path/to/experiment_results/section2/vllm \
  --labels_path /absolute/path/to/data/processed/section1/icu_cohort_data.csv \
  --output_root /absolute/path/to/experiment_results/section3 \
  --models llama \
  --run_demographics
```

## 4. Model Keys

Section 3 identifies files by the model key, not the full model path.

Examples:

- `llama`
- `deepseek`
- `gptoss120`

These must match the `--model_key` used in Section 2.

