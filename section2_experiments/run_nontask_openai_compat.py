#!/usr/bin/env python3

import argparse
import json
import math
import time
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm

from section2_utils import (
    get_input_representation,
    load_jsonl,
    metadata_from_item,
    resolve_output_path,
    write_jsonl,
)


def run_scoring(
    client: OpenAI,
    data: List[Dict[str, Any]],
    modality: str,
    include_demographics: bool,
    structured_format: str,
    output_path: str,
    batch_size: int,
    model_name: str,
    dataset_variant: str,
    sleep_s: float,
    model_key: str,
) -> None:
    results: List[Dict[str, Any]] = []

    for start in tqdm(range(0, len(data), batch_size), desc="Batches"):
        batch = data[start : start + batch_size]
        for item in batch:
            prompt = get_input_representation(
                item,
                modality=modality,
                include_demographics=include_demographics,
                structured_format=structured_format,
            )

            result = metadata_from_item(item)
            result.update(
                {
                    "experiment_type": "task_independent",
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
                    max_tokens=0,
                    temperature=0.0,
                    echo=True,
                    logprobs=1,
                )
                choice = response.choices[0]
                logprobs = getattr(choice, "logprobs", None)
                token_logprobs = getattr(logprobs, "token_logprobs", None) if logprobs is not None else None
                if not token_logprobs:
                    raise ValueError("Missing token_logprobs in completion response.")

                valid_logprobs: List[float] = []
                n_missing = 0
                for value in token_logprobs:
                    if value is None:
                        n_missing += 1
                    else:
                        valid_logprobs.append(float(value))
                if not valid_logprobs:
                    raise ValueError("All token_logprobs were None.")

                total_loglik = float(sum(valid_logprobs))
                n_tokens = int(len(valid_logprobs))
                avg_loglik = float(total_loglik / n_tokens)
                ppl = float(math.exp(-avg_loglik))

                result.update(
                    {
                        "n_tokens": n_tokens,
                        "log_likelihood_sum": total_loglik,
                        "log_likelihood_avg": avg_loglik,
                        "ppl": ppl,
                        "n_missing_logprobs": n_missing,
                        "prompt_n_chars": len(prompt),
                    }
                )
            except Exception as exc:
                result.update(
                    {
                        "n_tokens": None,
                        "log_likelihood_sum": None,
                        "log_likelihood_avg": None,
                        "ppl": None,
                        "error": str(exc),
                    }
                )

            results.append(result)
            if sleep_s > 0:
                time.sleep(sleep_s)

    write_jsonl(output_path, results)


def main():
    parser = argparse.ArgumentParser(description="Parameterized OpenAI-compatible runner for Section 2 task-independent experiments")
    parser.add_argument("--data", required=True, help="Input JSONL")
    parser.add_argument("--modality", required=True, choices=["raw", "template"])
    parser.add_argument("--output", default=None, help="Optional explicit output path")
    parser.add_argument("--output_dir", default=None, help="Directory used to derive output filename if --output is omitted")
    parser.add_argument("--dataset_variant", required=True, help="Dataset tag such as original, counterfactual, or demographics_cf")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model name exposed by the OpenAI-compatible server")
    parser.add_argument("--model_key", default=None, help="Short stable model key for filenames and downstream analysis")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--include_demographics", action="store_true", help="Include age/sex/race in structured template prompts")
    parser.add_argument("--structured_format", default="text", choices=["json", "text"], help="How to serialize template inputs")
    parser.add_argument("--sleep_s", type=float, default=0.0, help="Optional delay between requests")
    args = parser.parse_args()

    output_path = resolve_output_path(
        output=args.output,
        output_dir=args.output_dir,
        model_name=args.model,
        experiment_type="task_independent",
        modality=args.modality,
        dataset_variant=args.dataset_variant,
        include_demographics=args.include_demographics,
        model_key=args.model_key,
    )

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    data = load_jsonl(args.data)
    sliced_data = data[args.start : args.end] if args.end is not None else data[args.start :]

    print(json.dumps({"output_path": output_path, "backend": "openai_compat", "model": args.model, "n_rows": len(sliced_data)}, indent=2))
    started = time.time()

    run_scoring(
        client=client,
        data=sliced_data,
        modality=args.modality,
        include_demographics=args.include_demographics,
        structured_format=args.structured_format,
        output_path=output_path,
        batch_size=args.batch_size,
        model_name=args.model,
        dataset_variant=args.dataset_variant,
        sleep_s=args.sleep_s,
        model_key=args.model_key or args.model,
    )

    duration = time.time() - started
    print(f"Finished in {duration:.2f}s")


if __name__ == "__main__":
    main()
