###########################################################################
# Log-likelihood analysis for counterfactual notes 
# Input: JSONL files per model with log-likelihoods + shift_key JSON 
# Output:
# CSV with all log-likelihood deltas
# Plots:
#   a) Heatmap per model of Δ Log Likelihood vs class shift per vital
#   b) Line plots of Δ Log Likelihood vs counterfactual shift 
#   c) Summary barplot and table with aggregated values
###########################################################################

import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", required=True, help="Directory containing model result .jsonl files")
parser.add_argument("--shift_path", required=True, help="Path to shift_key_vitals.json")
parser.add_argument("--output_dir", default=".", help="Directory to save output plots and tables")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

model_files = {
    'obllm': "lkh_results_obllm.jsonl",
    'deepseek': "lkh_results_deepseek.jsonl",
    'phi': "lkh_results_phi.jsonl",
    'llama370': "lkh_results_llama370.jsonl",
    'meditron': "lkh_results_meditron.jsonl"
}
model_files = {k: os.path.join(args.results_dir, v) for k, v in model_files.items()}

with open(args.shift_path, 'r') as f:
    shift_key = json.load(f)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"], 
    "font.size": 18
})

hadm_to_original_class = {
    entry['hadm_id']: entry['original_class']
    for entries in shift_key.values()
    for entry in entries
}

def process_model(model_name, results_file):
    results = {}
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                results[entry['id']] = entry

    analysis = []

    for vital, entries in shift_key.items():
        for entry in entries:
            hadm_id = entry['hadm_id']
            original_id = f"{hadm_id}-original"

            if original_id not in results:
                continue

            original_ll_sum = results[original_id]['log_likelihood_sum']
            original_n_tokens = results[original_id]['n_tokens']

            for cf in entry['counterfactuals']:
                cf_id = cf['cf_id']
                if cf_id not in results:
                    continue

                cf_ll_sum = results[cf_id]['log_likelihood_sum']
                cf_n_tokens = results[cf_id]['n_tokens']

                delta_ll_sum = cf_ll_sum - original_ll_sum
                delta_ll_avg = (cf_ll_sum / cf_n_tokens) - (original_ll_sum / original_n_tokens)

                analysis.append({
                    'hadm_id': hadm_id,
                    'vital': vital,
                    'class_shift_diff': cf['diff'],
                    'abs_class_shift_diff': abs(cf['diff']),
                    'delta_log_likelihood_sum': delta_ll_sum,
                    'delta_log_likelihood_avg_per_token': delta_ll_avg,
                    'model': model_name
                })

    return pd.DataFrame(analysis)


all_models_df = []
for model_name, file_path in model_files.items():
    print(f"Processing {model_name}...")
    all_models_df.append(process_model(model_name, file_path))

full_df = pd.concat(all_models_df, ignore_index=True)
full_df["original_class"] = full_df["hadm_id"].map(hadm_to_original_class)

full_df.to_csv(os.path.join(args.output_dir, "all_models_lkh_analysis.csv"), index=False)


def analyze_and_plot(dataframe, suffix=""):
    agg_df = dataframe.groupby(['model', 'vital', 'class_shift_diff'])[['delta_log_likelihood_avg_per_token']].mean().reset_index()

    for model in model_files.keys():
        pivot_avg = agg_df[agg_df['model'] == model].pivot(index='vital', columns='class_shift_diff', values='delta_log_likelihood_avg_per_token')
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_avg, annot=True, cmap='coolwarm', center=0)
        plt.title(f"Delta Log Likelihood Avg per Vital - {model} {suffix}")
        plt.ylabel('Vital')
        plt.xlabel('Class Shift')
        plt.tight_layout()
        plt.savefig(f'heatmap_{model}{suffix}.png')
        plt.close()

    vitals = dataframe['vital'].unique()
    for vital in vitals:
        plt.figure(figsize=(10, 6))
        for model in model_files.keys():
            subset = agg_df[(agg_df['vital'] == vital) & (agg_df['model'] == model)]
            plt.plot(subset['class_shift_diff'], subset['delta_log_likelihood_avg_per_token'], label=model)
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f"Delta Log Likelihood vs Class Shift - {vital} {suffix}")
        plt.xlabel('Class Shift')
        plt.ylabel('Delta Log Likelihood (Avg per token)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'lineplot_{vital}{suffix}.png')
        plt.close()

    abs_df = dataframe.groupby(['model', 'vital', 'abs_class_shift_diff'])[['delta_log_likelihood_avg_per_token']].mean().reset_index()
    for vital in vitals:
        plt.figure(figsize=(10, 6))
        for model in model_files.keys():
            subset = abs_df[(abs_df['vital'] == vital) & (abs_df['model'] == model)]
            plt.plot(subset['abs_class_shift_diff'], subset['delta_log_likelihood_avg_per_token'], label=model)
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f"Delta Log Likelihood vs |Class Shift| - {vital} {suffix}")
        plt.xlabel('|Class Shift|')
        plt.ylabel('Delta Log Likelihood (Avg per token)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'abslineplot_{vital}{suffix}.png')
        plt.close()

    for model in model_files.keys():
        subset_df = dataframe[dataframe['model'] == model]
        grouped = subset_df.groupby(['vital', 'class_shift_diff'])[['delta_log_likelihood_avg_per_token']].mean().reset_index()

        plt.figure(figsize=(10, 6))
        for vital in grouped['vital'].unique():
            vital_data = grouped[grouped['vital'] == vital]
            plt.plot(vital_data['class_shift_diff'], vital_data['delta_log_likelihood_avg_per_token'], marker='o', label=vital)

        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f"{model}: Avg Log Likelihood Δ vs Class Shift (per Vital) {suffix}")
        plt.xlabel('Class Shift')
        plt.ylabel('Δ Log Likelihood (Avg per token)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{model}_vitals_loglik_vs_shift{suffix}.png')
        plt.close()

    summary_list = []
    for model_name in model_files.keys():
        model_data = dataframe[dataframe['model'] == model_name]
        overall_avg = model_data['delta_log_likelihood_avg_per_token'].mean()
        avg_shift_neg = model_data[model_data['class_shift_diff'] < 0]['delta_log_likelihood_avg_per_token'].mean()
        avg_shift_zero = model_data[model_data['class_shift_diff'] == 0]['delta_log_likelihood_avg_per_token'].mean()
        avg_shift_pos = model_data[model_data['class_shift_diff'] > 0]['delta_log_likelihood_avg_per_token'].mean()
        summary_list.append({
            'model': model_name,
            'overall_avg': overall_avg,
            'avg_shift_neg': avg_shift_neg,
            'avg_shift_zero': avg_shift_zero,
            'avg_shift_pos': avg_shift_pos
        })

    summary = pd.DataFrame(summary_list)
    summary.to_csv(f'summary_table_models{suffix}.csv', index=False)
    print(f"Summary for {suffix}:")
    print(summary)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=summary.melt(id_vars='model', value_vars=['overall_avg', 'avg_shift_neg', 'avg_shift_zero', 'avg_shift_pos']), 
                x='model', y='value', hue='variable')
    plt.title(f'Overall and Shift-specific Avg Delta Log Likelihood by Model {suffix}')
    plt.ylabel('Delta Log Likelihood (Avg per token)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'summary_barplot{suffix}.png')
    plt.close()

analyze_and_plot(full_df, suffix="")

#check only when original class is normal
normal_df = full_df[full_df['original_class'] == 'Normal']
analyze_and_plot(normal_df, suffix="_normal_only")

#signed severity
agg = (
    full_df
    .groupby(["model", "class_shift_diff"])
    .delta_log_likelihood_avg_per_token
    .agg(["mean", "std", "count"])
    .reset_index()
)
agg["se"] = agg["std"] / np.sqrt(agg["count"])
agg["ci95"] = 1.96 * agg["se"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11
})

palette = sns.color_palette("tab10", n_colors=agg.model.nunique())
fig, ax = plt.subplots(figsize=(10, 6))

for i, (model, g) in enumerate(agg.groupby("model")):
    c = palette[i]
    ax.plot(g["class_shift_diff"], g["mean"],
            label=model, color=c, marker="o", lw=1)
    ax.fill_between(g["class_shift_diff"],
                    g["mean"] - g["ci95"],
                    g["mean"] + g["ci95"],
                    color=c, alpha=0.15)

ax.axhline(0, ls="--", lw=1, c="grey")
ax.axvline(0, ls="--", lw=1, c="grey")
ax.set(
    title="Mean Δ Log Likelihood by Severity Shift per Model",
    xlabel="Counterfactual severity shift",
    ylabel="Δ Log Likelihood (avg per token)",
    xlim=(-4.5, 4.5)
)
ax.legend(title="Model", loc="best")
sns.despine()
plt.tight_layout()
plt.savefig("loglik_signed_severity_plot.png", dpi=300)
plt.savefig("loglik_signed_severity_plot.pdf", bbox_inches="tight", transparent=True)
plt.close()

print("\nOverall Δ Log Likelihood (avg per token) per model:")
model_stats = (
    full_df
    .groupby("model")["delta_log_likelihood_avg_per_token"]
    .agg(["mean", "std"])
    .reset_index()
    .rename(columns={"mean": "avg_loglik_change", "std": "std_loglik_change"})
)

print(model_stats.round(6).to_string(index=False))