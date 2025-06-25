##########################################################################################################
# Results analysis - raw notes and template-based notes. LOS task - zero-shot & fine-tuned models. 
# Needs jsonl files per model with the results & json shift_key file classifying couterfactuals severities
# Summary table per model with metrics (as in paper)
# Plot: boxplot distribution of JSD caused by each vital sign overall.
##########################################################################################################

import argparse
import json
import pandas as pd
import numpy as np
import ast
import os
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, required=True, help="Directory containing result JSONL files per model.")
parser.add_argument("--shift_path", type=str, required=True, help="Path to the shift_key JSON file.")
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output files.")
args = parser.parse_args()

def compute_jsd(p, q):
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    p /= np.sum(p)
    q /= np.sum(q)
    return jensenshannon(p, q, base=2) ** 2

def analyze_model_behavior(result_path, shift_path, model_name):
    with open(shift_path, 'r') as f:
        shift_data = json.load(f)

    cf_entries = []
    for vital_type, entries in shift_data.items():
        for item in entries:
            hadm_id = item['hadm_id']
            for cf in item['counterfactuals']:
                entry = {
                    'hadm_id': hadm_id,
                    'vital': vital_type,
                    **cf
                }
                cf_entries.append(entry)

    cf_df = pd.DataFrame(cf_entries)

    with open(result_path, 'r') as f:
        results = [json.loads(line) for line in f]
    results_df = pd.DataFrame(results)

    originals_df = results_df[results_df['id'].str.endswith('-original')].copy()
    counterfactuals_df = results_df[~results_df['id'].str.endswith('-original')].copy()
    
    def safe_parse(x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x

    originals_df["probs"] = originals_df["probs"].apply(safe_parse)
    cf_df = counterfactuals_df.merge(cf_df, left_on=['id', 'hadm_id'], right_on=['cf_id', 'hadm_id'], how='left')
    cf_df["probs"] = cf_df["probs"].apply(safe_parse)

    for k in ["1", "2", "3", "4"]:
        originals_df[f"p{k}_orig"] = originals_df["probs"].apply(lambda x: x.get(k, np.nan))
        cf_df[f"p{k}_cf"] = cf_df["probs"].apply(lambda x: x.get(k, np.nan))

    cf_merged = cf_df.merge(
        originals_df[["hadm_id"] + [f"p{k}_orig" for k in "1234"]],
        on="hadm_id",
        how="left"
    )

    p_orig = cf_merged[[f"p{k}_orig" for k in "1234"]].to_numpy()
    p_cf = cf_merged[[f"p{k}_cf" for k in "1234"]].to_numpy()

    cf_merged["abs_L1_shift"] = np.abs(p_cf - p_orig).sum(axis=1)
    cf_merged["js_divergence"] = [compute_jsd(p, q) for p, q in zip(p_orig, p_cf)]

    BUCKET_DAYS = np.array([3, 7, 14, 21])
    cf_merged["E_orig"] = p_orig @ BUCKET_DAYS
    cf_merged["E_cf"] = p_cf @ BUCKET_DAYS
    cf_merged["ΔE"] = cf_merged["E_cf"] - cf_merged["E_orig"]

    cf_merged["bucket_orig"] = np.argmax(p_orig, axis=1) + 1
    cf_merged["bucket_cf"] = np.argmax(p_cf, axis=1) + 1
    cf_merged["bucket_change"] = (cf_merged["bucket_orig"] != cf_merged["bucket_cf"]).astype(int)
    cf_merged["correct_direction"] = (np.sign(cf_merged["ΔE"]) == cf_merged["expected_sign"]).astype(int)

    summary_metrics = {}
    pos = cf_merged[cf_merged["ΔE"] > 0]
    neg = cf_merged[cf_merged["ΔE"] < 0]

    summary_metrics["model"] = model_name
    summary_metrics["%_ΔE_positive"] = len(pos) / len(cf_merged) * 100
    summary_metrics["avg_ΔE_positive"] = pos["ΔE"].mean() if not pos.empty else 0
    summary_metrics["std_ΔE_positive"] = pos["ΔE"].std() if not pos.empty else 0
    summary_metrics["%_ΔE_negative"] = len(neg) / len(cf_merged) * 100
    summary_metrics["avg_ΔE_negative"] = neg["ΔE"].mean() if not neg.empty else 0
    summary_metrics["std_ΔE_negative"] = neg["ΔE"].std() if not neg.empty else 0
    summary_metrics["avg_L1"] = cf_merged["abs_L1_shift"].mean()
    summary_metrics["std_L1"] = cf_merged["abs_L1_shift"].std()
    summary_metrics["avg_JSD"] = cf_merged["js_divergence"].mean()
    summary_metrics["std_JSD"] = cf_merged["js_divergence"].std()
    summary_metrics["%_correct_direction"] = cf_merged["correct_direction"].mean() * 100
    summary_metrics["%_hard_class_shift"] = cf_merged["bucket_change"].mean() * 100

    trend = (
        cf_merged.groupby("abs_sev")["ΔE"]
        .mean()
        .reset_index()
        .sort_values("abs_sev")       
        )
    good = bad = 0
    for i in range(len(trend) - 1):
        s1, de1 = trend.loc[i, ["abs_sev", "ΔE"]]
        s2, de2 = trend.loc[i + 1, ["abs_sev", "ΔE"]]

        if (s1 < 0 and s2 == 0) or (s1 == 0 and s2 > 0):
            continue                    

        if s1 < 0 and s2 < 0:            #negative side pair
            ok = (de2 < 0) and (de2 > de1)
        elif s1 > 0 and s2 > 0:          #positive side pair
            ok = (de2 > 0) and (de2 > de1)
        else:
            continue  

        if ok:
            good += 1
        else:
            bad += 1

    pct = good / (good + bad) * 100 if (good + bad) else np.nan
    summary_metrics["%_partial_magnitude_consistent4"] = pct  # %Mono in paper

    top_vitals = cf_merged.groupby("vital")["abs_L1_shift"].mean().sort_values(ascending=False).head(2).index.tolist()
    summary_metrics["top_vital_1"] = top_vitals[0] if len(top_vitals) > 0 else "N/A"
    summary_metrics["top_vital_2"] = top_vitals[1] if len(top_vitals) > 1 else "N/A"

    if "true_class" in originals_df.columns and "predicted_class" in originals_df.columns:
        summary_metrics["accuracy"] = accuracy_score(originals_df["true_class"], originals_df["predicted_class"])
        summary_metrics["f1"] = f1_score(originals_df["true_class"], originals_df["predicted_class"], average="macro")
        summary_metrics["qwk"] = cohen_kappa_score(originals_df["true_class"], originals_df["predicted_class"], weights="quadratic")
    else:
        summary_metrics["accuracy"] = "N/A"
        summary_metrics["f1"] = "N/A"
        summary_metrics["qwk"] = "N/A"

    cf_merged["model"] = model_name
    return summary_metrics, cf_merged[["vital", "abs_L1_shift", "js_divergence"]]


#results files
model_files = [
    "deepseekR1-ft", "llama370-ft", "phi4-ft", "obllm-ft",
    "meditron-0s", "deepseekR1-0s", "llama370-0s", "phi4-0s", "obllm-0s"
]

result_files = {
    name: os.path.join(args.results_dir, f"los_results_{name}.jsonl" if "0s" not in name else f"0s_results_{name.split('-')[0]}.jsonl")
    for name in model_files
}

all_summaries = []
all_vital_shifts = []

for model_name, result_path in result_files.items():
    print(f"Processing {model_name}...")
    summary, vital_data = analyze_model_behavior(result_path, args.shift_path, model_name)
    all_summaries.append(summary)
    all_vital_shifts.append(vital_data)

vital_df = pd.concat(all_vital_shifts, ignore_index=True)

#average L1 shift per vital across all models
avg_l1_per_vital = vital_df.groupby("vital")["abs_L1_shift"].mean().sort_values(ascending=False)

print("Average L1 shift per vital across all models:")
for vital, avg in avg_l1_per_vital.items():
    print(f"{vital}: {avg:.4f}")

summary_df = pd.DataFrame(all_summaries)
summary_csv_path = os.path.join(args.output_dir, "all_model_behavioral_summaries.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"\nSaved summary to {summary_csv_path}") 

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  
    "font.size": 18
})

vital_labels = {
    "blood_pressure": "BP",
    "heart_rate": "HR",
    "respiration_rate": "RR",
    "oxygen_saturation": "OxSat",
    "temperature": "Temp"
}

vital_df["vital_short"] = vital_df["vital"].map(vital_labels)

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=vital_df,
    x="vital_short",
    y="js_divergence",
    showfliers=False,
    color="#878A8F",
    width=0.6
)

plt.title("Distribution of JSD caused by each vital sign")
plt.xlabel("")
plt.ylabel("Jensen-Shannon Divergence (JSD)")
plt.tight_layout()
plt.savefig("boxplot_JSD_per_vital_notes.png", dpi=300)
plt.savefig("boxplot_JSD_per_vital_notes.pdf", bbox_inches="tight", transparent=True)
plt.show()