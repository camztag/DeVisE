#!/usr/bin/env python3

import argparse
import json
import math
import time
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from section2_utils import (
    build_task_instruction,
    extract_label,
    get_input_representation,
    load_jsonl,
    metadata_from_item,
    parse_los_bins,
    resolve_output_path,
    write_jsonl,
)


def build_gptoss_prompt(
    tokenizer: Any,
    task: str,
    modality: str,
    input_text: str,
    los_bins,
) -> str:
    content_label = "Structured admission note (JSON):" if modality == "template" else "Patient admission note:"
    instruction = build_task_instruction(task, bins=los_bins)
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"{content_label}\n{input_text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + "[["


def extract_probs_from_top_logprobs(top_logprobs: Dict[str, float], labels: List[int]) -> Dict[int, float]:
    raw_scores: Dict[int, float] = {}
    for label in labels:
        raw_scores[label] = -1e9

    for token, logprob in top_logprobs.items():
        clean = str(token).strip().replace("▁", "").replace("Ġ", "").replace("Â", "")
        if clean.isdigit():
            label = int(clean)
            if label in raw_scores:
                raw_scores[label] = float(logprob)

    max_score = max(raw_scores.values())
    exp_scores = {label: math.exp(raw_scores[label] - max_score) for label in raw_scores}
    norm = sum(exp_scores.values())
    if norm <= 0:
        return {label: None for label in labels}
    return {label: exp_scores[label] / norm for label in labels}


def run_predictions(
    client: OpenAI,
    tokenizer: Any,
    data: List[Dict[str, Any]],
    task: str,
    modality: str,
    include_demographics: bool,
    structured_format: str,
    output_path: str,
    batch_size: int,
    los_bins,
    model_name: str,
    dataset_variant: str,
    sleep_s: float,
    model_key: str,
) -> None:
    class_labels = [0, 1] if task == "mortality" else [1, 2, 3, 4]
    results: List[Dict[str, Any]] = []

    for start in tqdm(range(0, len(data), batch_size), desc="Batches"):
        batch = data[start : start + batch_size]
        for item in batch:
            prompt = build_gptoss_prompt(
                tokenizer=tokenizer,
                task=task,
                input_text=get_input_representation(
                    item,
                    modality=modality,
                    include_demographics=include_demographics,
                    structured_format=structured_format,
                ),
                los_bins=los_bins,
                modality=modality,
            )

            result = metadata_from_item(item)
            result.update(
                {
                    "experiment_type": task,
                    "modality": modality,
                    "dataset_variant": dataset_variant,
                    "include_demographics": include_demographics if modality == "template" else None,
                    "model_name": model_name,
                    "model_key": model_key,
                }
            )

            try:
                response = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=3,
                    temperature=0.0,
                    logprobs=20,
                    echo=False,
                    stop=["]]"],
                )
                choice = response.choices[0]
                raw_output = choice.text
                pred_text = extract_label(raw_output if raw_output.startswith("[[") else f"[[{raw_output}", task=task)

                probs = {label: None for label in class_labels}
                pred_prob_class = None
                if choice.logprobs and choice.logprobs.top_logprobs and len(choice.logprobs.top_logprobs) > 0:
                    probs = extract_probs_from_top_logprobs(choice.logprobs.top_logprobs[0], class_labels)
                    finite_probs = {label: prob for label, prob in probs.items() if prob is not None}
                    if finite_probs:
                        pred_prob_class = max(finite_probs, key=finite_probs.get)

                result.update(
                    {
                        "pred_text": pred_text,
                        "pred_prob_class": pred_prob_class,
                        "raw_output": raw_output,
                    }
                )
                for label in class_labels:
                    result[f"prob_{label}"] = probs.get(label)

            except Exception as exc:
                result.update(
                    {
                        "pred_text": None,
                        "pred_prob_class": None,
                        "raw_output": None,
                        "error": str(exc),
                    }
                )
                for label in class_labels:
                    result[f"prob_{label}"] = None

            results.append(result)
            if sleep_s > 0:
                time.sleep(sleep_s)

    write_jsonl(output_path, results)


def main():
    parser = argparse.ArgumentParser(description="Parameterized OpenAI-compatible runner for Section 2 task experiments")
    parser.add_argument("--data", required=True, help="Input JSONL")
    parser.add_argument("--task", required=True, choices=["mortality", "los"])
    parser.add_argument("--modality", required=True, choices=["raw", "template"])
    parser.add_argument("--output", default=None, help="Optional explicit output path")
    parser.add_argument("--output_dir", default=None, help="Directory used to derive output filename if --output is omitted")
    parser.add_argument("--dataset_variant", required=True, help="Dataset tag such as original, counterfactual, or demographics_cf")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model name exposed by the OpenAI-compatible server")
    parser.add_argument("--model_key", default=None, help="Short stable model key for filenames and downstream analysis")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--prompt_type", default="llama", help="Ignored for GPT-OSS tokenizer-template prompting.")
    parser.add_argument("--tokenizer_name", default=None, help="Tokenizer to use for GPT-OSS chat template. Defaults to --model.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--include_demographics", action="store_true", help="Include age/sex/race in structured template prompts")
    parser.add_argument("--structured_format", default="json", choices=["json", "text"], help="How to serialize template inputs")
    parser.add_argument("--los_bins", default=None, help="LOS bin spec like '24:38,39:59,60:112,113:657'")
    parser.add_argument("--cohort_csv", default=None, help="Optional Section 1 cohort CSV used to derive LOS quartiles when --los_bins is omitted")
    parser.add_argument("--sleep_s", type=float, default=0.0, help="Optional delay between requests")
    args = parser.parse_args()

    output_path = resolve_output_path(
        output=args.output,
        output_dir=args.output_dir,
        model_name=args.model,
        experiment_type=args.task,
        modality=args.modality,
        dataset_variant=args.dataset_variant,
        include_demographics=args.include_demographics,
        model_key=args.model_key,
    )

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    data = load_jsonl(args.data)
    sliced_data = data[args.start : args.end] if args.end is not None else data[args.start :]
    los_bins = parse_los_bins(args.los_bins, cohort_path=args.cohort_csv) if args.task == "los" else None
    tokenizer_name = args.tokenizer_name or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print(json.dumps({"output_path": output_path, "backend": "openai_compat", "model": args.model, "tokenizer": tokenizer_name, "n_rows": len(sliced_data)}, indent=2))
    started = time.time()

    run_predictions(
        client=client,
        tokenizer=tokenizer,
        data=sliced_data,
        task=args.task,
        modality=args.modality,
        include_demographics=args.include_demographics,
        structured_format=args.structured_format,
        output_path=output_path,
        batch_size=args.batch_size,
        los_bins=los_bins,
        model_name=args.model,
        dataset_variant=args.dataset_variant,
        sleep_s=args.sleep_s,
        model_key=args.model_key or args.model,
    )

    duration = time.time() - started
    print(f"Finished in {duration:.2f}s")


if __name__ == "__main__":
    main()
