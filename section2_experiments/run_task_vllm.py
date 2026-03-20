#!/usr/bin/env python3

import argparse
import json
import time
from typing import Any, Dict, List

from tqdm import tqdm
from vllm import LLM, SamplingParams

from section2_utils import (
    build_task_prompt,
    extract_class_probabilities,
    extract_label,
    get_input_representation,
    get_vllm_profile,
    load_jsonl,
    metadata_from_item,
    parse_los_bins,
    resolve_output_path,
    write_jsonl,
)


def run_predictions(
    llm: LLM,
    data: List[Dict[str, Any]],
    task: str,
    modality: str,
    prompt_type: str,
    include_demographics: bool,
    structured_format: str,
    output_path: str,
    batch_size: int,
    los_bins,
    model_name: str,
    dataset_variant: str,
    model_key: str,
) -> None:
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    class_labels = [0, 1] if task == "mortality" else [1, 2, 3, 4]

    results: List[Dict[str, Any]] = []

    for start in tqdm(range(0, len(data), batch_size), desc="Batches"):
        batch = data[start : start + batch_size]
        prompts = [
            build_task_prompt(
                prompt_type=prompt_type,
                task=task,
                input_text=get_input_representation(
                    item,
                    modality=modality,
                    include_demographics=include_demographics,
                    structured_format=structured_format,
                ),
                bins=los_bins,
                modality=modality,
                structured_format=structured_format,
            )
            for item in batch
        ]
        outputs = llm.generate(prompts, sampling_params=params)

        for item, output in zip(batch, outputs):
            generated = output.outputs[0]
            raw_output = generated.text
            pred_text = extract_label(raw_output, task=task)
            pred_prob_class, probs = extract_class_probabilities(generated.logprobs, tokenizer, class_labels)

            result = metadata_from_item(item)
            result.update(
                {
                    "experiment_type": task,
                    "modality": modality,
                    "dataset_variant": dataset_variant,
                    "include_demographics": include_demographics if modality == "template" else None,
                    "model_name": model_name,
                    "model_key": model_key,
                    "pred_text": pred_text,
                    "pred_prob_class": pred_prob_class,
                    "raw_output": raw_output,
                }
            )
            for label in class_labels:
                result[f"prob_{label}"] = probs.get(label)
            results.append(result)

    write_jsonl(output_path, results)


def main():
    parser = argparse.ArgumentParser(description="Parameterized vLLM runner for Section 2 task experiments")
    parser.add_argument("--data", required=True, help="Input JSONL")
    parser.add_argument("--task", required=True, choices=["mortality", "los"])
    parser.add_argument("--modality", required=True, choices=["raw", "template"])
    parser.add_argument("--output", default=None, help="Optional explicit output path")
    parser.add_argument("--output_dir", default=None, help="Directory used to derive output filename if --output is omitted")
    parser.add_argument("--dataset_variant", required=True, help="Dataset tag such as original, counterfactual, or demographics_cf")
    parser.add_argument("--model_path", required=True, help="Model name or local path for vLLM")
    parser.add_argument("--model_key", default=None, help="Short stable model key for filenames and downstream analysis")
    parser.add_argument("--prompt_type", required=True, choices=["llama", "deepseek", "phi", "qwen"])
    parser.add_argument("--model_profile", default="default_1gpu", help="Profile name from section2_utils.VLLM_MODEL_PROFILES")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--include_demographics", action="store_true", help="Include age/sex/race in structured template prompts")
    parser.add_argument("--structured_format", default="json", choices=["json", "text"], help="How to serialize template inputs")
    parser.add_argument("--los_bins", default=None, help="LOS bin spec like '24:38,39:59,60:112,113:657'")
    parser.add_argument("--cohort_csv", default=None, help="Optional Section 1 cohort CSV used to derive LOS quartiles when --los_bins is omitted")
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--dtype", default=None, choices=["float16", "bfloat16"])
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    args = parser.parse_args()

    if args.modality == "raw" and args.include_demographics:
        print("[INFO] --include_demographics is ignored for raw modality.")

    output_path = resolve_output_path(
        output=args.output,
        output_dir=args.output_dir,
        model_name=args.model_path,
        experiment_type=args.task,
        modality=args.modality,
        dataset_variant=args.dataset_variant,
        include_demographics=args.include_demographics,
        model_key=args.model_key,
    )

    profile = get_vllm_profile(
        args.model_profile,
        {
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "trust_remote_code": args.trust_remote_code if args.trust_remote_code else None,
            "enforce_eager": args.enforce_eager if args.enforce_eager else None,
        },
    )

    data = load_jsonl(args.data)
    sliced_data = data[args.start : args.end] if args.end is not None else data[args.start :]
    los_bins = parse_los_bins(args.los_bins, cohort_path=args.cohort_csv) if args.task == "los" else None

    llm = LLM(model=args.model_path, **profile)

    print(json.dumps({"output_path": output_path, "profile": profile, "n_rows": len(sliced_data)}, indent=2))
    started = time.time()

    run_predictions(
        llm=llm,
        data=sliced_data,
        task=args.task,
        modality=args.modality,
        prompt_type=args.prompt_type,
        include_demographics=args.include_demographics,
        structured_format=args.structured_format,
        output_path=output_path,
        batch_size=args.batch_size,
        los_bins=los_bins,
        model_name=args.model_path,
        dataset_variant=args.dataset_variant,
        model_key=args.model_key or args.model_path,
    )

    duration = time.time() - started
    print(f"Finished in {duration:.2f}s")


if __name__ == "__main__":
    main()
