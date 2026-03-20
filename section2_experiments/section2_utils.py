#!/usr/bin/env python3

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math

import pandas as pd

VLLM_MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
    "default_1gpu": {
        "tensor_parallel_size": 1,
        "dtype": "float16",
        "max_model_len": 4096,
        "enforce_eager": True,
        "trust_remote_code": False,
    },
    "default_2gpu": {
        "tensor_parallel_size": 2,
        "dtype": "float16",
        "max_model_len": 4096,
        "enforce_eager": True,
        "trust_remote_code": False,
    },
    "gptoss_1gpu": {
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "enforce_eager": True,
        "trust_remote_code": True,
    },
    "gptoss_2gpu": {
        "tensor_parallel_size": 2,
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "enforce_eager": True,
        "trust_remote_code": True,
    },
}

METADATA_KEYS = [
    "hadm_id",
    "subject_id",
    "id",
    "original_class",
    "original_severity",
    "counterfactual_class",
    "counterfactual_severity",
    "class_diff",
    "class_diff_abs",
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_parent_dir(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_value(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, str):
        if value.strip() == "" or value.lower() in {"nan", "none"}:
            return "Unknown"
        return value
    if isinstance(value, float):
        if value != value:
            return "Unknown"
    return str(value)


def build_structured_payload(item: Dict[str, Any], include_demographics: bool = True) -> Dict[str, Any]:
    vitals = item.get("vitals") or {}
    payload: Dict[str, Any] = {
        "vitals": {
            "temperature": vitals.get("temperature"),
            "heart_rate": vitals.get("heart_rate"),
            "blood_pressure": vitals.get("blood_pressure"),
            "respiration_rate": vitals.get("respiration_rate"),
            "oxygen_saturation": vitals.get("oxygen_saturation"),
        }
    }
    if include_demographics:
        payload["age"] = item.get("age")
        payload["sex"] = item.get("sex")
        payload["race"] = item.get("race")
    return payload


def build_structured_text(item: Dict[str, Any], include_demographics: bool = True) -> str:
    parts: List[str] = []
    if include_demographics:
        parts.append(f"Age: {format_value(item.get('age'))}")
        parts.append(f"Sex: {format_value(item.get('sex'))}")
        parts.append(f"Race: {format_value(item.get('race'))}")

    vitals = item.get("vitals") or {}
    vitals_parts = [
        f"Temperature: {format_value(vitals.get('temperature'))}",
        f"Heart Rate: {format_value(vitals.get('heart_rate'))}",
        f"Blood Pressure: {format_value(vitals.get('blood_pressure'))}",
        f"Respiration Rate: {format_value(vitals.get('respiration_rate'))}",
        f"Oxygen Saturation: {format_value(vitals.get('oxygen_saturation'))}",
    ]
    parts.append("Vitals - " + ", ".join(vitals_parts))
    return " | ".join(parts)


def get_input_representation(
    item: Dict[str, Any],
    modality: str,
    include_demographics: bool = True,
    structured_format: str = "text",
) -> str:
    if modality == "raw":
        return item["text"]

    # Templates should always include demographics + vitals
    if structured_format == "json":
        return json.dumps(build_structured_payload(item, include_demographics=True), ensure_ascii=False)
    return build_structured_text(item, include_demographics=True)


def normalize_prompt_type(prompt_type: str) -> str:
    prompt_type = prompt_type.lower()
    if prompt_type not in {"llama", "deepseek", "phi", "qwen"}:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return prompt_type


def compute_los_quantiles_from_cohort(cohort_path: str) -> Tuple[float, float, float]:
    df = pd.read_csv(cohort_path, low_memory=False)
    los_col = "episode_los_hours" if "episode_los_hours" in df.columns else "los_icu_hours"
    if los_col not in df.columns:
        raise ValueError(f"{cohort_path} must contain episode_los_hours or los_icu_hours.")
    los = pd.to_numeric(df[los_col], errors="coerce").dropna()
    if los.empty:
        raise ValueError(f"No usable LOS values found in {cohort_path}.")
    q1, q2, q3 = [float(x) for x in los.quantile([0.25, 0.50, 0.75]).tolist()]
    return q1, q2, q3


def build_los_bins_from_quantiles(q1: float, q2: float, q3: float) -> List[Tuple[int, float, float]]:
    return [
        (1, 24.0, q1),
        (2, q1, q2),
        (3, q2, q3),
        (4, q3, q3),
    ]


def parse_los_bins(bin_spec: Optional[str], cohort_path: Optional[str] = None) -> List[Tuple[int, float, float]]:
    if not bin_spec:
        if cohort_path:
            q1, q2, q3 = compute_los_quantiles_from_cohort(cohort_path)
            return build_los_bins_from_quantiles(q1, q2, q3)
        raise ValueError("LOS runs require either --los_bins or --cohort_csv.")

    bins: List[Tuple[int, float, float]] = []
    for idx, chunk in enumerate(bin_spec.split(","), start=1):
        lo_str, hi_str = chunk.split(":")
        bins.append((idx, float(lo_str), float(hi_str)))
    return bins


def _fmt_hours(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def los_bins_text(bins: List[Tuple[int, float, float]]) -> str:
    lines = []
    for label, lo, hi in bins:
        descriptor = {
            1: "Very short stay",
            2: "Short stay",
            3: "Moderate stay",
            4: "Long stay",
        }.get(label, f"Bucket {label}")
        if label == 4 and abs(hi - lo) < 1e-9:
            lines.append(f"[[{label}]] {descriptor} ({_fmt_hours(lo)}+ hours).")
        else:
            lines.append(f"[[{label}]] {descriptor} ({_fmt_hours(lo)} to {_fmt_hours(hi)} hours).")
    return "\n".join(lines)


def build_task_instruction(task: str, bins: Optional[List[Tuple[int, float, float]]] = None) -> str:
    if task == "mortality":
        return (
            "You are an expert in ICU mortality prediction.\n"
            "Based only on the patient's admission information, predict whether the patient will die before ICU discharge.\n\n"
            "Return only the target value in double brackets:\n"
            "[[0]] for survival\n"
            "[[1]] for death."
        )

    if task == "los":
        if not bins:
            raise ValueError("LOS task instruction requires explicit bins derived from --los_bins or --cohort_csv.")
        los_bins = bins
        return (
            "You are an expert in ICU length-of-stay prediction.\n"
            "Based only on the patient's admission information, predict in which ICU length-of-stay bucket the patient will fall.\n\n"
            "We divide ICU length of stay (for stays >= 24 hours) into 4 groups:\n"
            f"{los_bins_text(los_bins)}\n"
            "Return only the bucket in double brackets: [[1]], [[2]], [[3]], or [[4]]."
        )

    raise ValueError(f"Unknown task: {task}")


def build_task_prompt(
    prompt_type: str,
    task: str,
    input_text: str,
    bins: Optional[List[Tuple[int, int, int]]] = None,
    modality: str = "raw",
    structured_format: str = "json",
) -> str:
    prompt_type = normalize_prompt_type(prompt_type)
    if modality == "template":
        content_label = "Structured admission note (JSON):" if structured_format == "json" else "Structured admission note:"
    else:
        content_label = "Patient admission note:"
    instruction = build_task_instruction(task, bins=bins)

    if prompt_type == "llama":
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{instruction}\n"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{content_label}\n{input_text}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            "[["
        )

    if prompt_type == "deepseek":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{instruction}\n\n"
            f"{content_label}\n{input_text}\n"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            "[["
        )

    if prompt_type == "phi":
        return (
            "<|im_start|>system<|im_sep|>\n"
            f"{instruction}\n"
            "<|im_end|>\n"
            "<|im_start|>user<|im_sep|>\n"
            f"{content_label}\n{input_text}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant<|im_sep|>\n"
            "[["
        )

    return (
        "<|im_start|>system\n"
        f"You are Qwen, created by Alibaba Cloud. {instruction}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{content_label}\n{input_text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "[["
    )


def extract_label(text: str, task: str) -> Optional[int]:
    text = (
        (text or "")
        .translate(_DIGIT_TRANSLATION)
        .strip()
        .replace("▁", "")
        .replace("Ġ", "")
        .replace("Â", "")
    )

    allowed = {"0", "1"} if task == "mortality" else {"1", "2", "3", "4"}

    # Main case for your current setup: model returns just the digit
    if text in allowed:
        return int(text)

    # Fallback in case a future setup returns the full bracketed form
    pattern = r"\[\[\s*([0-9])\s*\]\]"
    match = re.search(pattern, text)
    if match and match.group(1) in allowed:
        return int(match.group(1))

    return None


_DIGIT_TRANSLATION = str.maketrans(
    "₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹",
    "01234567890123456789"
)

def extract_class_probabilities(logprobs_list: Any, tokenizer: Any, labels: List[int]) -> Tuple[Optional[int], Dict[int, Optional[float]]]:
    if not logprobs_list:
        return None, {label: None for label in labels}

    first_step = logprobs_list[0]
    raw_scores: Dict[int, Optional[float]] = {label: None for label in labels}

    for token_id, payload in first_step.items():
        token_str = tokenizer.decode([token_id])

        clean = (
            token_str
            .translate(_DIGIT_TRANSLATION)
            .strip()
            .replace("▁", "")
            .replace("Ġ", "")
            .replace("Â", "")
        )

        # old-script behavior: only trust exact clean class tokens
        if clean not in {str(label) for label in labels}:
            continue

        label = int(clean)
        raw_scores[label] = payload.logprob if hasattr(payload, "logprob") else payload

    if all(raw_scores[label] is None for label in labels):
        return None, {label: None for label in labels}

    filled = {label: (raw_scores[label] if raw_scores[label] is not None else -1e9) for label in labels}
    max_score = max(filled.values())
    exp_scores = {label: math.exp(filled[label] - max_score) for label in labels}
    norm = sum(exp_scores.values())

    if norm <= 0 or not math.isfinite(norm):
        return None, {label: None for label in labels}

    probs = {label: exp_scores[label] / norm for label in labels}
    pred_class = max(probs, key=probs.get)
    return pred_class, probs

def metadata_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for key in METADATA_KEYS:
        if key in item:
            meta[key] = item.get(key)
    return meta


def slugify_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-").lower()


def build_output_filename(
    model_name: str,
    experiment_type: str,
    modality: str,
    dataset_variant: str,
    include_demographics: bool,
    model_key: Optional[str] = None,
) -> str:
    model_slug = slugify_model_name(model_key or model_name)
    demographics_tag = "with_demographics" if include_demographics else "no_demographics"
    return f"{model_slug}__{experiment_type}__{modality}__{dataset_variant}__{demographics_tag}.jsonl"


def resolve_output_path(
    output: Optional[str],
    output_dir: Optional[str],
    model_name: str,
    experiment_type: str,
    modality: str,
    dataset_variant: str,
    include_demographics: bool,
    model_key: Optional[str] = None,
) -> str:
    if output:
        return output
    if not output_dir:
        raise ValueError("Either --output or --output_dir must be provided.")
    return str(Path(output_dir) / build_output_filename(model_name, experiment_type, modality, dataset_variant, include_demographics, model_key=model_key))


def get_vllm_profile(profile_name: Optional[str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    profile = dict(VLLM_MODEL_PROFILES.get(profile_name or "default_1gpu", {}))
    if not profile:
        raise ValueError(f"Unknown model profile: {profile_name}")
    for key, value in overrides.items():
        if value is not None:
            profile[key] = value
    return profile
