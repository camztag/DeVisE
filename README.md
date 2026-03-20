# DeVisE: Behavioral Testing of Medical Large Language Models

Code repository for the paper:

**DeVisE: Behavioral Testing of Medical Large Language Models**  
Camila Zurdo Tagliabue, Heloisa Oss Boll, Aykut Erdem, Erkut Erdem, Iacer Calixto

This repository supports the full DeVisE pipeline:

1. prepare ICU data and construct controlled counterfactual datasets
2. run zero-shot experiments on original and counterfactual inputs
3. analyze behavioral sensitivity and downstream prediction shifts

DeVisE evaluates whether medical LLMs respond to clinically meaningful perturbations, rather than relying on superficial correlations. Using ICU discharge notes from MIMIC-IV, the pipeline constructs both raw and template-based examples, perturbs demographics and vital signs in controlled ways, and measures the resulting changes in perplexity, mortality predictions, and length-of-stay predictions.

## Paper

DeVisE: Towards the Behavioral Testing of Medical Large Language Models  
OpenReview: https://openreview.net/forum?id=n68TTKPzbk

## Overview

The repository is organized into three stages:

1. `section1_counterfactuals`: build the cohort, clean notes, extract vitals, and generate counterfactual datasets
2. `section2_experiments`: run zero-shot experiments on original and counterfactual inputs
3. `section3_behavioral_evaluation`: analyze model sensitivity and downstream prediction shifts

Pipeline flow:

- Section 1 -> produces datasets for Section 2
- Section 2 -> produces model outputs for Section 3
- Section 3 -> analyzes Section 2 outputs

## Repository Layout

Recommended working layout:

```text
Repository-26/
├── README.md
├── project_context.md
├── run_section1.sh
├── data/
│   ├── raw/                          # downloaded MIMIC files
│   │   ├── icustays.csv
│   │   ├── admissions.csv
│   │   ├── patients.csv
│   │   └── discharge.csv
│   └── processed/
│       └── section1/                 # Section 1 intermediate and final data products
├── experiment_results/
│   ├── section2/                     # Section 2 outputs from all backends and models
│   └── section3/
│       ├── mortality/
│       ├── los/
│       ├── task_independent/
│       └── demographics/
├── plots/                            # optional exported figures from Section 3
├── section1_counterfactuals/
├── section2_experiments/
├── section3_behavioral_evaluation/
└── ...
```

The code does not force this exact structure, but it is the recommended organization for reproducible runs because it separates:

- raw source data
- processed data
- experiment outputs
- analysis outputs and plots
- code and documentation


## Sections

### Section 1

Data preparation and counterfactual generation.

Main outputs:

- original unstructured admission notes
- counterfactual unstructured notes
- original structured template notes
- counterfactual structured template notes
- demographic counterfactual structured template notes

See:

- [`section1_counterfactuals/README_section1.md`](section1_counterfactuals/README_section1.md)
- [`run_section1.sh`](run_section1.sh)

### Section 2

Zero-shot experiments on:

- raw vs template inputs
- mortality vs LOS vs task-independent scoring
- original vs counterfactual vs demographics counterfactual inputs
- vLLM backend vs OpenAI-compatible / GPT-OSS backend

See:

- [`section2_experiments/README_section2.md`](section2_experiments/README_section2.md)
- [`section2_experiments/run_section2.sh`](section2_experiments/run_section2.sh)

### Section 3

Behavioral analyses over Section 2 outputs:

- mortality robustness analyses
- LOS robustness analyses
- task-independent perplexity shift analyses
- demographics counterfactual analyses

See:

- [`section3_behavioral_evaluation/README_section3.md`](section3_behavioral_evaluation/README_section3.md)
- [`section3_behavioral_evaluation/run_section3.sh`](section3_behavioral_evaluation/run_section3.sh)

## Quick Start

### 1. Prepare directories

```bash
mkdir -p data/raw data/processed/section1
mkdir -p experiment_results/section2 experiment_results/section3
mkdir -p plots
```

### 2. Run Section 1

```bash
bash ./run_section1.sh \
  --raw-data-dir ./data/raw \
  --output-dir ./data/processed/section1
```

### 3. Run Section 2

vLLM example:

```bash
bash './section2_experiments/run_section2.sh' \
  --input_root ./data/processed/section1 \
  --output_root ./experiment_results/section2 \
  --backend vllm \
  --model_path 'meta-llama/Llama-3.3-70B-Instruct' \
  --model_key llama \
  --prompt_type llama \
  --model_profile default_2gpu
```

OpenAI-compatible / GPT-OSS example:

```bash
bash './section2_experiments/run_section2.sh' \
  --input_root ./data/processed/section1 \
  --output_root ./experiment_results/section2 \
  --backend openai_compat \
  --model 'openai/gpt-oss-120b' \
  --model_key gptoss120 \
  --base_url 'http://localhost:8000/v1' \
  --api_key 'EMPTY'
```

### 4. Run Section 3

Example Section 3 run:

```bash
bash './section3_behavioral_evaluation/run_section3.sh' \
  --results_dir ./experiment_results/section2 \
  --labels_path ./data/processed/section1/icu_cohort_data.csv \
  --output_root ./experiment_results/section3 \
  --models llama
```

## Environments

This repository currently assumes two execution environments:

### Environment A: standard vLLM

Use for most Section 1 and Section 2 model runs.

Install with:

```bash
pip install -r requirements.txt
```

### Environment B: GPT-OSS / OpenAI-compatible serving

Use for the OpenAI-compatible Section 2 runs if GPT-OSS requires a different serving setup or vLLM version.

Install with:

```bash
pip install -r requirements-gpt.txt
```

## Models

The repository is backend-agnostic and supports both local and OpenAI-compatible models.

Example model keys used in the DeVisE experiments include:
`llama`, `obllm`, `phi`, `meditron`, `deepseek`, `gptoss120`, and `qwen25`.

These keys are used to name Section 2 outputs and retrieve them in Section 3.

## Notes on Data Access

This project uses MIMIC-IV and MIMIC-IV-Note, which require credentialed PhysioNet access.

See the Section 1 README for download details and required files.


## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{tagliabue2026devise,
  title = {{DeVisE: Towards the Behavioral Testing of Medical Large Language Models}},
  author = {Tagliabue, Camila Zurdo and Boll, Heloisa Oss and Erdem, Aykut and Erdem, Erkut and Calixto, Iacer},
  booktitle = {19th Conference of the European Chapter of the Association for Computational Linguistics},
  year = {2026},
  url = {https://openreview.net/forum?id=n68TTKPzbk},
}
```
