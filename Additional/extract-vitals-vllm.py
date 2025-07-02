import json
import re
import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm

def load_notes_from_jsonl(file_path):
    print(f"Loading notes from {file_path}")
    with open(file_path, as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def fix_content_quotes(json_text):
    def escape_inner_quotes(match):
        prefix, content, suffix = match.group(1), match.group(2), match.group(3)
        fixed_content = re.sub(r'(?<!\\\)"', r'\\"', content)
        return prefix + fixed_content + suffix
    return re.sub(r'("content":\s*")(.*?)(")', escape_inner_quotes, json_text, flags=re.DOTALL)

def extract_first_json(text):
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found")
    brace_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                candidate = text[start:i+1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue
    raise ValueError("Incomplete JSON object found")

def clean_generated_text(text):
    print(f"Generated text before cleaning: {text}")
    text = re.sub(r"```(?:json)?", "", text)
    cleaned_text = text.strip()
    print(f"Generated text after cleaning: {cleaned_text}")
    return cleaned_text

def build_vitals_extraction_prompt(section_content, subject_id, hamd_id):
    example_input = {
        "subject_id": "12345",
        "hamd_id": "abcde",
        "section": "PHYSICAL EXAM",
        "content": "Vitals: T 97.7, HR 110, BP 99/62, RR 25, O2 99%"
    }
    example_output = {
        "subject_id": "12345",
        "hamd_id": "abcde",
        "vitals": {
            "temperature": "97.7",
            "heart_rate": "110",
            "blood_pressure": "99/62",
            "respiration_rate": "25",
            "oxygen_saturation": "99%"
        }
    }

    example_input2 = {
        "subject_id": "12347",
        "hamd_id": "abcde",
        "section": "PHYSICAL EXAM",
        "content": "T 98.2, P 117 ,O2 97%"
    }
    example_output2 = {
        "subject_id": "12345",
        "hamd_id": "abcde",
        "vitals": {
            "temperature": "97.7",
            "heart_rate": "117",
            "blood_pressure": "",
            "respiration_rate": "",
            "oxygen_saturation": "97%"
        }
    }

    prompt = f"""
    You are a clinical information extraction assistant. Your task is to extract the vitals from the "PHYSICAL EXAM" section of a clinical note.

    - Temperature
    - Heart Rate (or Pulse)
    - Blood Pressure
    - Respiration Rate
    - Oxygen Saturation

    The vitals text may present these values in various formats.

    Example Input 1:
    {json.dumps(example_input, indent=2)}

    Example Output 1:
    {json.dumps(example_output, indent=2)}

    Example Input 2:
    {json.dumps(example_input2, indent=2)}

    Example Output 2:
    {json.dumps(example_output2, indent=2)}

    IMPORTANT: Return ONLY the JSON object and nothing else.

    Input:
    {{
      "subject_id": "{subject_id}",
      "hamd_id": "{hamd_id}",
      "section": "PHYSICAL EXAM",
      "content": "{section_content.strip()}"
    }}

    Output:
    <<<JSON OUTPUT>>>
    """
    return prompt

def process_llm_batch(llm, prompts, metadata, sampling_params):
    print(f"Processing batch of size {len(prompts)}")
    results = []
    outputs = llm.generate(prompts, sampling_params)
    print(f"LLM Outputs: {outputs}")

    for out, meta in zip(outputs, metadata):
        try:
            text = clean_generated_text(out.outputs[0].text)
            if "<<<JSON OUTPUT>>>" in text:
                text = text.split("<<<JSON OUTPUT>>>", 1)[1]
            json_part = extract_first_json(text)
            fixed = fix_content_quotes(json_part)
            parsed = json.loads(fixed)

            vitals = parsed.get("vitals", {})
            for key in ["temperature", "heart_rate", "blood_pressure", "respiration_rate", "oxygen_saturation"]:
                if not vitals.get(key, "").strip():
                    vitals[key] = "NaN"

            results.append({
                "subject_id": meta["subject_id"],
                "hamd_id": meta["hamd_id"],
                "vitals": vitals
            })

        except Exception as e:
            print(f"[ERROR] Failed to parse result for subject_id {meta['subject_id']}: {e}")
            print(f"Output was: {out.outputs[0].text}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Vitals Extraction with vLLM")
    parser.add_argument("--input_file", type=str, required=True, help=".jsonl file with input notes")
    parser.add_argument("--output_file", type=str, required=True, help="Path to write output .jsonl file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for vLLM calls")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of notes to process.")
    args = parser.parse_args()

    llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct",
              dtype="float16", tensor_parallel_size=2, max_model_len=4896, enforce_eager=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=800)

    batch_prompts = []
    batch_metadata = []
    all_results = []

    note_count = 0
    total_to_process = args.limit if args.limit else float('inf')
    pbar = tqdm(total=total_to_process, desc="Processing notes")

    for note in load_notes_from_jsonl(args.input_file):
        print(f"Processing note {note_count + 1}")
        if note_count >= total_to_process:
            break

        subject_id = note.get("subject_id", "")
        hamd_id = note.get("hamd_id", "")
        section = next((s for s in note.get("notes", []) if s.get("section") == "PHYSICAL EXAM"), None)

        if section:
            prompt = build_vitals_extraction_prompt(section.get("content", ""), subject_id, hamd_id)
            batch_prompts.append(prompt)
            batch_metadata.append({"subject_id": subject_id, "hamd_id": hamd_id})

            if len(batch_prompts) >= args.batch_size:
                all_results.extend(process_llm_batch(llm, batch_prompts, batch_metadata, sampling_params))
                batch_prompts, batch_metadata = [], []
        note_count += 1
        pbar.update(1)

        if note_count % 10 == 0:
            print(f"[INFO] Processed {note_count} notes...", flush=True)
        
    if batch_prompts:
        all_results.extend(process_llm_batch(llm, batch_prompts, batch_metadata, sampling_params))

    print(f"Writing results to {args.output_file}")
    with open(args.output_file, "w") as f:
        for item in all_results:
            f.write(json.dumps(item) + "\n")
    pbar.close()
    print(f"[DONE] Extracted vitals written to {args.output_file}")

if __name__ == "__main__":
    main()