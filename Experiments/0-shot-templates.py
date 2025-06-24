#########################################################################################
# Zero-shot prompts for Length of Stay task on template-based notes. 
# To be used with models with a Llama 3, Phi4, or DeepSeek R1 base. 
# Obtains the probability of the first 20 higher probs tokens (including the LOS bucket)
# Extracts the probs of each LOS bucket and normalizes them.
#########################################################################################

import json
import os
import time
import argparse
from typing import List, Dict
from tqdm import tqdm
from vllm import LLM, SamplingParams
import math

def load_data(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: List[Dict], path: str):
    """Save a list of dicts to a JSONL file, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def format_prompt_default(record: Dict) -> str:
    """Format a template record into the prompt body."""
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

def format_prompt(template_name: str, record: Dict) -> str:
    instruction = (
        "You are an expert in hospital stay predictions. Predict the patient's total length of ICU stay "
        "based on their admission note summary. Output only a number between double brackets: [[1]] for <= 3 days, "
        "[[2]] for >3 & <=7 days, [[3]] for >1 and <=2 weeks, [[4]] for  >2 weeks."
    )

    if template_name == "default":
        return (
            "You are an expert in hospital stay predictions. Predict the patient's total length of ICU stay "
            "based on their admission note summary. Output only a number between double brackets: [[1]] for <= 3 days, "
            "[[2]] for >3 & <=7 days, [[3]] for >1 and <=2 weeks, [[4]] for  >2 weeks.\n\n"
            f"{format_prompt_default(record)}\n\n"
            "Answer: [["
        )

    note_text = format_prompt_default(record)
    input_text = f"Note:\n{note_text}"

    if template_name == "llama3":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{instruction}\n\n{input_text}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n[["
        )
    elif template_name == "phi4":
        return (
            "<|im_start|>user<|im_sep|>"
            f"{instruction}\n\n{input_text}"
            "<|im_end|>\n<|im_start|>assistant<|im_sep|>\n[["
        )
    elif template_name == "deepseekr1":
        return (
            "<｜begin_of_sentence｜><｜User｜>\n"
            f"{instruction}\n\n{input_text}\n"
            "<｜Assistant｜>\n[["
        )
    else:
        raise ValueError(f"Unknown template: {template_name}")

def predict_bucket_probs(llm, data, output_path, batch_size=32, template_name="default"):
    """Generate bucket probabilities for each sample and write results to output_path."""
    results = []
    tokenizer = llm.get_tokenizer()
    target_tokens = ["1", "2", "3", "4"]

    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]

        prompts = []
        for item in batch:
            prompt = format_prompt(template_name, item)
            prompts.append(prompt)

        params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=20,
        )

        outputs = llm.generate(prompts, sampling_params=params)

        for item, output in zip(batch, outputs):
            pred_probs = {token: 0.0 for token in target_tokens}

            if not output.outputs:
                print(f"[WARNING] No output for hadm_id {item.get('hadm_id')}, skipping...")
                continue

            out = output.outputs[0]
            if out.logprobs is None or len(out.logprobs) == 0:
                print(f"[WARNING] No logprobs for hadm_id {item.get('hadm_id')}, skipping...")
                continue

            logprobs_dict = out.logprobs[0]
            print("Top logprobs for hadm_id", item.get("hadm_id"), "→", logprobs_dict)

            for tk, lp in logprobs_dict.items():
                tok_str = tokenizer.decode([tk]) if isinstance(tk, int) else tk
                clean_tok = tok_str.strip().replace("Ġ", "").replace("▁", "")
                if clean_tok in target_tokens:
                    raw_logp = lp.logprob if hasattr(lp, 'logprob') else lp
                    pred_probs[clean_tok] = math.exp(raw_logp)

            total = sum(pred_probs.values())
            if total > 0:
                pred_probs = {k: v / total for k, v in pred_probs.items()}
            else:
                pred_probs = {k: 0.25 for k in pred_probs}

            predicted_class = max(pred_probs, key=pred_probs.get)
            true_class = str(item.get("los_class"))
            is_correct = 1 if predicted_class == true_class else 0

            results.append({
                "hadm_id": item.get("hadm_id"),
                "subject_id": item.get("subject_id"),
                "id": item.get("id") or f"{item['hadm_id']}-original",
                "probs": pred_probs,
                "predicted_class": predicted_class,
                "true_class": true_class,
                "is_correct": is_correct,
            })

    save_jsonl(results, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to input JSONL data file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL results")
    parser.add_argument("--model_path", type=str, required=True, help="Path to vLLM model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prompt generation")
    parser.add_argument("--start", type=int, default=0, help="Starting index of data slice")
    parser.add_argument("--end", type=int, default=None, help="Ending index of data slice")
    parser.add_argument("--template", type=str, choices=["default", "llama3", "phi4", "deepseekr1"], default="default")


    args = parser.parse_args()

    llm = LLM(
        model=args.model_path,
        dtype="float16",
        tensor_parallel_size=2,
        max_model_len=1054,
        enforce_eager=True,
    )

    data = load_data(args.data)
    sliced_data = data[args.start:args.end] if args.end is not None else data[args.start:]

    print(f"Running {len(sliced_data)} notes using template '{args.template}'...")
    print(f"Starting run for {len(sliced_data)} templates with batch size {args.batch_size}")
    start_time = time.time()
    predict_bucket_probs(llm, sliced_data, args.output, batch_size=args.batch_size, template_name=args.template)
    end_time = time.time()

    total_time = end_time - start_time
    seconds_per_note = total_time / len(sliced_data)

    print("\nFinished run.")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Seconds per template: {seconds_per_note:.4f} s/template")
