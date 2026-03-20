#!/usr/bin/env python3

import argparse
import json
import time
from typing import Any, Dict, List

from tqdm import tqdm
from vllm import LLM, SamplingParams

from section2_utils import (
    get_input_representation,
    get_vllm_profile,
    load_jsonl,
    metadata_from_item,
    resolve_output_path,
    write_jsonl,
)


def run_scoring(
    llm: LLM,
    data: List[Dict[str, Any]],
    modality: str,
    include_demographics: bool,
    structured_format: str,
    output_path: str,
    batch_size: int,
    max_model_len: int,
    model_name: str,
    dataset_variant: str,
    model_key: str,
) -> None:
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, logprobs=0)
    results: List[Dict[str, Any]] = []

    for start in tqdm(range(0, len(data), batch_size), desc="Batches"):
        batch = data[start : start + batch_size]
        prompts = [
            get_input_representation(
                item,
                modality=modality,
                include_demographics=include_demographics,
                structured_format=structured_format,
            )
            for item in batch
        ]
        outputs = llm.generate(prompts, sampling_params=params)

        for item, output, prompt_text in zip(batch, outputs, prompts):
            prompt_lp = getattr(output, "prompt_logprobs", None)
            prompt_ids = getattr(output, "prompt_token_ids", None)
            if not prompt_lp or not prompt_ids:
                continue

            token_logprobs: List[float] = []
            n_missing = 0
            n_special = 0
            for token_id, entry in zip(prompt_ids, prompt_lp):
                if entry is None:
                    n_special += 1
                    continue
                lp_obj = entry.get(token_id)
                if lp_obj is None:
                    n_missing += 1
                    continue
                token_logprobs.append(lp_obj.logprob)

            if not token_logprobs:
                continue

            total_loglik = float(sum(token_logprobs))
            n_tokens = int(len(token_logprobs))
            avg_loglik = float(total_loglik / n_tokens)
            ppl = float(pow(2.718281828459045, -avg_loglik))

            result = metadata_from_item(item)
            result.update(
                {
                    "experiment_type": "task_independent",
                    "modality": modality,
                    "dataset_variant": dataset_variant,
                    "include_demographics": include_demographics if modality == "template" else None,
                    "model_name": model_name,
                    "model_key": model_key,
                    "n_tokens": n_tokens,
                    "log_likelihood_sum": total_loglik,
                    "log_likelihood_avg": avg_loglik,
                    "ppl": ppl,
                    "n_special_positions": n_special,
                    "n_missing_logprobs": n_missing,
                    "was_truncated_heuristic": bool(len(prompt_ids) >= max_model_len),
                    "prompt_n_chars": len(prompt_text),
                }
            )
            results.append(result)

    write_jsonl(output_path, results)


def main():
    parser = argparse.ArgumentParser(description="Parameterized vLLM runner for Section 2 task-independent experiments")
    parser.add_argument("--data", required=True, help="Input JSONL")
    parser.add_argument("--modality", required=True, choices=["raw", "template"])
    parser.add_argument("--output", default=None, help="Optional explicit output path")
    parser.add_argument("--output_dir", default=None, help="Directory used to derive output filename if --output is omitted")
    parser.add_argument("--dataset_variant", required=True, help="Dataset tag such as original, counterfactual, or demographics_cf")
    parser.add_argument("--model_path", required=True, help="Model name or local path for vLLM")
    parser.add_argument("--model_key", default=None, help="Short stable model key for filenames and downstream analysis")
    parser.add_argument("--model_profile", default="default_1gpu", help="Profile name from section2_utils.VLLM_MODEL_PROFILES")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--include_demographics", action="store_true", help="Include age/sex/race in structured template prompts")
    parser.add_argument("--structured_format", default="text", choices=["json", "text"], help="How to serialize template inputs")
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
        experiment_type="task_independent",
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
    llm = LLM(model=args.model_path, **profile)

    print(json.dumps({"output_path": output_path, "profile": profile, "n_rows": len(sliced_data)}, indent=2))
    started = time.time()

    run_scoring(
        llm=llm,
        data=sliced_data,
        modality=args.modality,
        include_demographics=args.include_demographics,
        structured_format=args.structured_format,
        output_path=output_path,
        batch_size=args.batch_size,
        max_model_len=profile["max_model_len"],
        model_name=args.model_path,
        dataset_variant=args.dataset_variant,
        model_key=args.model_key or args.model_path,
    )

    duration = time.time() - started
    print(f"Finished in {duration:.2f}s")


if __name__ == "__main__":
    main()
