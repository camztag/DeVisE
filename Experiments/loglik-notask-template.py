#########################################################################################
# Inputs template-based notes, sums logprobs for all tokens, and calculateds average per.
# note.
# Outputs per note # tokens, sum logprobs & average.
# 
#########################################################################################

import json
import os
import time
from typing import List, Dict
from vllm import LLM, SamplingParams
from tqdm import tqdm


def load_data(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def format_prompt(record: Dict) -> str:
    vit = record["vitals"]
    return (
        f"age: {record['age']}\n"
        f"sex: {record['sex']}\n"
        f"race: {record['race']}\n"
        f"vitals: "
        f"temperature: {vit['temperature']}, "
        f"heart_rate: {vit['heart_rate']}, "
        f"blood_pressure: {vit['blood_pressure']}, "
        f"respiration_rate: {vit['respiration_rate']}, "
        f"oxygen_saturation: {vit['oxygen_saturation']}"
    )


def loglik_mode(llm, data, output_path, batch_size=32):
    results = []

    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        prompts = [item["text"] for item in batch]

        params = SamplingParams(
            logprobs=0,
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
        )

        outputs = llm.generate(prompts, sampling_params=params)

        for item, output in zip(batch, outputs):
            prompt_lp = output.prompt_logprobs
            if not prompt_lp:
                print(f"[WARNING] No prompt_logprobs for hadm_id {item.get('hadm_id')}, skipping…")
                continue

            chosen_lps = []
            for entry in prompt_lp:
                if entry is None:
                    continue
                if isinstance(entry, list):
                    lp_obj = entry[-1]
                else:
                    lp_obj = next(
                        (cand for cand in entry.values() if cand.rank == 1),
                        list(entry.values())[-1]
                    )
                chosen_lps.append(lp_obj)

            logprobs = [lp.logprob for lp in chosen_lps]
            if not logprobs:
                print(f"[WARNING] Empty logprobs list for hadm_id {item.get('hadm_id')}, skipping…")
                continue

            total_loglik = sum(logprobs)
            avg_loglik = total_loglik / len(logprobs)

            results.append({
                "hadm_id": item.get("hadm_id"),
                "subject_id": item.get("subject_id"),
                "id": item.get("id") or f"{item['hadm_id']}-original",
                "n_tokens": len(logprobs),
                "log_likelihood_sum": total_loglik,
                "log_likelihood_avg": avg_loglik,
            })

    save_jsonl(results, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/projects/0/prjs1270/data/templates/counterfactuals/demo_counterfactuals_merged.jsonl")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    args = parser.parse_args()

    llm = LLM(
        model=args.model_path,
        dtype="float16",
        tensor_parallel_size=2,
        max_model_len=3000,
        enforce_eager=True,
    )

    data = load_data(args.data)

    for item in data:
        item["text"] = format_prompt(item)

    sliced_data = data[args.start:args.end] if args.end is not None else data[args.start:]

    print(f"Starting run for {len(sliced_data)} notes with batch size {args.batch_size}")
    start_time = time.time()
    loglik_mode(llm, sliced_data, args.output, batch_size=args.batch_size)
    end_time = time.time()

    total_time = end_time - start_time
    seconds_per_note = total_time / len(sliced_data)
    print(f"\nFinished run.")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Seconds per note: {seconds_per_note:.4f} s/note")
