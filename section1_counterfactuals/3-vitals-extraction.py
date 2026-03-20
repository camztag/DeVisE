#!/usr/bin/env python3
import json
import re
import argparse
from typing import Dict, Any, Generator, List

from vllm import LLM, SamplingParams
from tqdm import tqdm


# ------------------------------------------------------
# Load notes from .jsonl
# Expected input schema per line:
# {"subject_id": ..., "hadm_id": ..., "text": "..."}
# ------------------------------------------------------
def load_notes_from_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
    print(f"[INFO] Loading notes from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSON on line {line_num}: {e}")


# ------------------------------------------------------
# Extract PHYSICAL EXAM from cleaned note text
# ------------------------------------------------------
_PHYS_RE = re.compile(
    r"PHYSICAL EXAM:\s*(.*?)(?:\n\nFAMILY HISTORY:|\n\nSOCIAL HISTORY:|$)",
    flags=re.IGNORECASE | re.DOTALL,
)

def extract_physical_exam(text: str) -> str:
    if not text:
        return ""
    m = _PHYS_RE.search(text)
    if not m:
        return ""
    return m.group(1).strip()


# ------------------------------------------------------
# Fix unescaped quotes inside the "content" value
# ------------------------------------------------------
def fix_content_quotes(json_text: str) -> str:
    def escape_inner_quotes(match):
        prefix, content, suffix = match.group(1), match.group(2), match.group(3)
        fixed_content = re.sub(r'(?<!\\)"', r'\\"', content)
        return prefix + fixed_content + suffix

    return re.sub(r'("content":\s*")(.*?)(")', escape_inner_quotes, json_text, flags=re.DOTALL)


# ------------------------------------------------------
# Extract first complete JSON object from text
# ------------------------------------------------------
def extract_first_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                candidate = text[start:i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

    raise ValueError("Incomplete JSON object found in model output")


# ------------------------------------------------------
# Remove code fences / trim text
# ------------------------------------------------------
def clean_generated_text(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    return text.strip()


# ------------------------------------------------------
# Prompt
# ------------------------------------------------------
def build_vitals_extraction_prompt(physical_exam_content: str, subject_id: int, hadm_id: int) -> str:
    example_input_1 = {
        "subject_id": "12345",
        "hadm_id": "67890",
        "section": "PHYSICAL EXAM",
        "content": "Vitals: T 97.7, HR 110, BP 99/62, RR 25, O2 99%"
    }
    example_output_1 = {
        "subject_id": "12345",
        "hadm_id": "67890",
        "vitals": {
            "temperature": "97.7",
            "heart_rate": "110",
            "blood_pressure": "99/62",
            "respiration_rate": "25",
            "oxygen_saturation": "99%"
        }
    }

    example_input_2 = {
        "subject_id": "12346",
        "hadm_id": "67891",
        "section": "PHYSICAL EXAM",
        "content": "T 98.2, P 117, O2 98%."
    }
    example_output_2 = {
        "subject_id": "12346",
        "hadm_id": "67891",
        "vitals": {
            "temperature": "98.2",
            "heart_rate": "117",
            "blood_pressure": "",
            "respiration_rate": "",
            "oxygen_saturation": "98%"
        }
    }

    example_input_3 = {
    "subject_id": "12347",
    "hadm_id": "67892",
    "section": "PHYSICAL EXAM",
    "content": "General: patient awake, alert, in no acute distress. HEENT: moist mucous membranes. Lungs clear."
    }

    example_output_3 = {
        "subject_id": "12347",
        "hadm_id": "67892",
        "vitals": {
            "temperature": "",
            "heart_rate": "",
            "blood_pressure": "",
            "respiration_rate": "",
            "oxygen_saturation": ""
        }
    }

    safe_content = physical_exam_content.replace("\\", "\\\\").replace('"', '\\"')

    prompt = f"""
You are a clinical information extraction assistant. Your task is to extract the vitals from the "PHYSICAL EXAM" section of a clinical note.

The vitals to extract are:
- Temperature
- Heart Rate (or Pulse)
- Blood Pressure
- Respiration Rate
- Oxygen Saturation

The physical exam text may present these values in various formats. For example:
- "T: 97.6, P: 114, BP: 93/62, RR: 16, O2 98% on RA"
- "VS: T 90.9, HR 120, BP 99/55, RR 25, O2 94%"
- "Vitals: 99.2    69   119/79   25   96%"
- "VS: Temp: 96.2 (Tm 97.3), BP: 178/80, HR: 72, RR: 19, O2 sat: 99%"
- "T:98.1 BP:130/70 P:93 R:18 O2:96"
- "98.1 130/70 93 18 96%RA"
- For unlabeled values such as "98.1 130/70 18 96%RA", if the third number is less than 40, interpret it as the respiration rate; otherwise, interpret it as the heart rate.

Please extract the vitals from the provided text.
Return the result as a JSON object with the following structure exactly (without any extra text or markdown formatting):

{{
  "subject_id": "{int(subject_id)}",
  "hadm_id": "{int(hadm_id)}",
  "vitals": {{
    "temperature": "",
    "heart_rate": "",
    "blood_pressure": "",
    "respiration_rate": "",
    "oxygen_saturation": ""
  }}
}}

If a vital sign is not present in the text, leave its value as an empty string.

Do not infer, guess, estimate, or complete missing values.
Do not use typical normal values.
Do not copy values from the examples.
Only extract values that are explicitly written in the input text.

If no vitals are present at all, return the JSON object with all empty strings for the vitals.

Example Input 1:
{json.dumps(example_input_1, indent=2)}

Example Output 1:
{json.dumps(example_output_1, indent=2)}

Example Input 2:
{json.dumps(example_input_2, indent=2)}

Example Output 2:
{json.dumps(example_output_2, indent=2)}

Example Input 3:
{json.dumps(example_input_3, indent=2)}

Example Output 3:
{json.dumps(example_output_3, indent=2)}

IMPORTANT: Return ONLY the JSON object and nothing else.

Input:
{{
  "subject_id": "{int(subject_id)}",
  "hadm_id": "{int(hadm_id)}",
  "section": "PHYSICAL EXAM",
  "content": "{safe_content}"
}}

Output:
<<<JSON OUTPUT>>>
""".strip()

    return prompt


# ------------------------------------------------------
# Normalize output vitals
# ------------------------------------------------------
VITAL_KEYS = ["temperature", "heart_rate", "blood_pressure", "respiration_rate", "oxygen_saturation"]

def normalize_vitals(vitals: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for key in VITAL_KEYS:
        value = vitals.get(key, "")
        value = "" if value is None else str(value).strip()
        out[key] = value if value else "NaN"
    return out


# ------------------------------------------------------
# Process a batch through vLLM
# ------------------------------------------------------
def process_llm_batch(
    llm: LLM,
    prompts: List[str],
    metadata: List[Dict[str, Any]],
    sampling_params: SamplingParams,
) -> List[Dict[str, Any]]:
    print(f"[INFO] Processing batch of size {len(prompts)}")
    results: List[Dict[str, Any]] = []

    try:
        outputs = llm.generate(prompts, sampling_params)
    except Exception as e:
        print(f"[ERROR] Failed to generate output for this batch: {e}")
        with open("failed_batch_prompts.jsonl", "a", encoding="utf-8") as debug_file:
            for meta, prompt in zip(metadata, prompts):
                debug_file.write(json.dumps({
                    "subject_id": meta["subject_id"],
                    "hadm_id": meta["hadm_id"],
                    "prompt": prompt
                }, ensure_ascii=False) + "\n")
        return results

    for out, meta in zip(outputs, metadata):
        raw_text = out.outputs[0].text if out.outputs else ""
        try:
            cleaned = clean_generated_text(raw_text)

            if "<<<JSON OUTPUT>>>" in cleaned:
                cleaned = cleaned.split("<<<JSON OUTPUT>>>", 1)[1]

            json_part = extract_first_json(cleaned)
            fixed = fix_content_quotes(json_part)
            parsed = json.loads(fixed)

            vitals = normalize_vitals(parsed.get("vitals", {}))

            results.append({
                "subject_id": int(meta["subject_id"]),
                "hadm_id": int(meta["hadm_id"]),
                "vitals": vitals
            })

        except Exception as e:
            print(
                f"[ERROR] Failed to parse output for subject_id={meta.get('subject_id')} "
                f"hadm_id={meta.get('hadm_id')}: {e}"
            )
            print(f"[ERROR] Raw output was:\n{raw_text}\n")

            results.append({
                "subject_id": int(meta["subject_id"]),
                "hadm_id": int(meta["hadm_id"]),
                "vitals": {k: "NaN" for k in VITAL_KEYS},
                "parse_error": True
            })

    return results


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Vitals extraction from PHYSICAL EXAM with vLLM")
    parser.add_argument("--input_file", type=str, required=True, help="Input .jsonl with {subject_id, hadm_id, text}")
    parser.add_argument("--output_file", type=str, required=True, help="Output .jsonl with extracted vitals")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for vLLM calls")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of notes to process")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="vLLM model name/path")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--max_model_len", type=int, default=4896, help="Max model length")
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=800)

    batch_prompts: List[str] = []
    batch_metadata: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    note_count = 0
    total_to_process = args.limit if args.limit is not None else None
    pbar = tqdm(total=total_to_process, desc="Processing notes")

    for note in load_notes_from_jsonl(args.input_file):
        if total_to_process is not None and note_count >= total_to_process:
            break

        subject_id = note.get("subject_id")
        hadm_id = note.get("hadm_id")
        text = note.get("text", "")

        if subject_id is None or hadm_id is None:
            note_count += 1
            pbar.update(1)
            continue

        phys = extract_physical_exam(text)
        if not phys:
            print(f"[WARN] No PHYSICAL EXAM extracted for subject_id={subject_id} hadm_id={hadm_id}")
            note_count += 1
            pbar.update(1)
            continue

        prompt = build_vitals_extraction_prompt(phys, int(subject_id), int(hadm_id))
        batch_prompts.append(prompt)
        batch_metadata.append({
            "subject_id": int(subject_id),
            "hadm_id": int(hadm_id)
        })

        if len(batch_prompts) >= args.batch_size:
            all_results.extend(process_llm_batch(llm, batch_prompts, batch_metadata, sampling_params))
            batch_prompts, batch_metadata = [], []

        note_count += 1
        pbar.update(1)

        if note_count % 10 == 0:
            tqdm.write(f"[INFO] Processed {note_count} notes...")

    if batch_prompts:
        all_results.extend(process_llm_batch(llm, batch_prompts, batch_metadata, sampling_params))

    print(f"[INFO] Writing results to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    pbar.close()
    print(f"[DONE] Extracted vitals written to {args.output_file}")
    print(f"[DONE] Total processed input rows: {note_count}")
    print(f"[DONE] Total output rows written: {len(all_results)}")


if __name__ == "__main__":
    main()