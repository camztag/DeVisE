##########################################################################################################
# Raw notes results analysis. LOS task - zero-shot & fine-tuned models. 
# Needs jsonl files per model with the results & json shift_key file classifying couterfactuals severities
# Full counterfactual analysis all models - per counterfactual - data frame
# Plots: a) Mean ΔE[LOS] by severity per model
#        b) Jensen–Shannon Divergence (JSD) boxpolot per model
# Group-wise average JSD comparisons - ttests & wilcoxon signed-rank test
##########################################################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np
import ast
from scipy.stats import ttest_rel, wilcoxon
from scipy.spatial.distance import jensenshannon

full_data = []

result_files = {
    "dsR1(FT)": "/Users/camila.tagliabue@goflink.com/OneDrive - UvA/SRP/results/los_results_deepseekR1.jsonl",
    "dsR1(ZS)": "/Users/camila.tagliabue@goflink.com/OneDrive - UvA/SRP/results/0s_results_deepseekR1.jsonl",
    "llama3(FT)": "/Users/camila.tagliabue@goflink.com/OneDrive - UvA/SRP/results/los_results_llama370.jsonl",
    "llama3(ZS)": "/Users/camila.tagliabue@goflink.com/OneDrive - UvA/SRP/results/0s_results_llama370.jsonl",
    "phi4(FT)": "/Users/camila.tagliabue@goflink.com/OneDrive - UvA/SRP/results/los_results_phi4.jsonl",
    "phi4(ZS)": "/Users/camila.tagliabue@goflink.com/OneDrive - UvA/SRP/results/0s_results_phi4.jsonl",
    "obllm(FT)": "/Users/camila.tagliabue@goflink.com/OneDrive - UvA/SRP/results/los_results_obllm.jsonl",
    "obllm(ZS)": "/Users/camila.tagliabue@goflink.com/OneDrive - UvA/SRP/results/0s_results_obllm.jsonl",
    "meditron(ZS)": "/Users/camila.tagliabue@goflink.com/OneDrive - UvA/SRP/results/0s_results_meditron.jsonl"
}

shift_path = "/Users/camila.tagliabue@goflink.com/OneDrive - UvA/SRP/results/shift_key2.json"

for model_name, result_path in result_files.items():
    print(f"Processing {model_name} for full dataframe...")
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

    def compute_jsd(p, q):
        p = np.clip(p, 1e-12, 1)
        q = np.clip(q, 1e-12, 1)
        p /= np.sum(p)
        q /= np.sum(q)
        return jensenshannon(p, q, base=2) ** 2

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
        on="hadm_id", how="left"
    )

    p_orig = cf_merged[[f"p{k}_orig" for k in "1234"]].to_numpy()
    p_cf = cf_merged[[f"p{k}_cf" for k in "1234"]].to_numpy()

    cf_merged["js_divergence"] = [compute_jsd(p, q) for p, q in zip(p_orig, p_cf)]

    BUCKET_DAYS = np.array([3, 7, 14, 21])
    cf_merged["E_orig"] = p_orig @ BUCKET_DAYS
    cf_merged["E_cf"] = p_cf @ BUCKET_DAYS
    cf_merged["ΔE"] = cf_merged["E_cf"] - cf_merged["E_orig"]

    cf_merged["model"] = model_name
    full_data.append(cf_merged[["hadm_id", "abs_sev", "ΔE", "abs_L1_shift", "js_divergence", "model"]])

df_all = pd.concat(full_data, ignore_index=True)
df_all.to_csv("full_counterfactual_analysis_all_models.csv", index=False)

#---------------------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"], 
    "font.size": 18
})

agg = (df_all
       .groupby(["model", "abs_sev"])
       .ΔE.agg(["mean", "std", "count"])
       .reset_index())
agg["se"]   = agg["std"] / np.sqrt(agg["count"])
agg["ci95"] = 1.96 * agg["se"]  

palette = sns.color_palette("tab10", n_colors=agg.model.nunique())
fig, ax   = plt.subplots(figsize=(10,6))
for i, (model, g) in enumerate(agg.groupby("model")):
    c = palette[i]
    ax.plot( g.abs_sev, g["mean"],
             label=model, color=c, marker="o", markersize=3, lw=1 )
    ax.fill_between( g.abs_sev,
                     g["mean"]-g.ci95, g["mean"]+g.ci95,
                     color=c, alpha=0.15, zorder=-1 )
    
ax.axhline(0, ls="--", lw=1, c="grey")
ax.axvline(0, ls="--", lw=1, c="grey")

ax.set(
    title = "Mean ΔE[LOS] by severity per model",
    ylabel= "ΔE[LOS] (days)",
    xlabel= "Counterfactual severity shift",
    xlim  = (-4, 4),
)
ax.spines[["top", "right"]].set_visible(False)   
sns.despine()                                   

plt.legend(
    title="Model",
    loc="lower right",
    bbox_to_anchor=(1, 0),  
    ncol=2,
    frameon=True,
    borderpad=0.4,
    fontsize="medium",
    labelspacing=0.3,
    columnspacing=0.5
)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig("dE_severity_per_model.png", dpi=300)
plt.savefig("dE_severity_per_model.pdf",
            bbox_inches="tight", 
            transparent=True) 
plt.close()

#------------------------------------------------------------------------------------
#JSD per model
plt.figure(figsize=(10, 6))

ax = sns.boxplot(
    data=df_all,
    x="model",
    y="js_divergence",
    showfliers=False,        
    color="#878A8F",           
    width=0.6
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_ylabel("Jensen–Shannon Divergence (JSD)")
ax.set_xlabel("")
ax.set_title("Overall JSD per Model")

plt.tight_layout()

plt.savefig("boxplot_JSD_per_model2.png", dpi=300)
plt.savefig("boxplot_JSD_per_model2.pdf", bbox_inches="tight", transparent=True)
plt.close()

#----------------------------------------------------------------------------------------
#group-wise average JSD comparisons

#model groups
ft_models = [m for m in df_all["model"].unique() if "(FT)" in m]
zs_models = [m for m in df_all["model"].unique() if "(ZS)" in m]


ft_groups = {
    "Medical FT": ["obllm(FT)"],
    "GP FT": ["phi4(FT)", "llama3(FT)"],
    "Thinking FT": ["dsR1(FT)"],
}

zs_groups = {
    "Medical ZS": ["obllm(ZS)", "meditron(ZS)"],
    "GP ZS": ["phi4(ZS)", "llama3(ZS)"],
    "Thinking ZS": ["dsR1(ZS)"],
}

def compute_group_jsd(df, model_list):
    return (
        df[df["model"].isin(model_list)]
        .groupby(["hadm_id", "abs_sev"])["js_divergence"]
        .mean()
    )

group_jsd_ft = {name: compute_group_jsd(df_all, models) for name, models in ft_groups.items()}
group_jsd_zs = {name: compute_group_jsd(df_all, models) for name, models in zs_groups.items()}

def format_p(p):
    if p < 1e-300:
        return "<1e-300"
    elif p < 0.001:
        return f"{p:.1e}"
    else:
        return f"{p:.3f}"

#function to compare all pairs
def compare_groups_jsd(group_dict):
    stats_summary = []
    keys = list(group_dict.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            g1, g2 = keys[i], keys[j]
            jsd1, jsd2 = group_dict[g1], group_dict[g2]
            jsd1, jsd2 = jsd1.align(jsd2, join="inner")

            t_stat, t_p = ttest_rel(jsd1, jsd2)
            w_stat, w_p = wilcoxon(jsd1, jsd2)

            stats_summary.append({
                "Comparison": f"{g1} vs {g2}",
                f"Mean {g1}": jsd1.mean(),
                f"Std {g1}": jsd1.std(),
                f"Mean {g2}": jsd2.mean(),
                f"Std {g2}": jsd2.std(),
                "Paired t p-value": format_p(t_p),
                "Wilcoxon p-value": format_p(w_p),
            })
    return pd.DataFrame(stats_summary)

#run comparisons and print results
df_ft_jsd_stats = compare_groups_jsd(group_jsd_ft)
df_zs_jsd_stats = compare_groups_jsd(group_jsd_zs)

print("\n=== Fine-Tuned Model Group Comparisons (JSD) ===")
print(df_ft_jsd_stats.to_string(index=False))

print("\n=== Zero-Shot Model Group Comparisons (JSD) ===")
print(df_zs_jsd_stats.to_string(index=False))

mean_jsd_ft = (
    df_all[df_all["model"].isin(ft_models)]
    .groupby(["hadm_id", "abs_sev"])["js_divergence"]
    .mean()
)

mean_jsd_zs = (
    df_all[df_all["model"].isin(zs_models)]
    .groupby(["hadm_id", "abs_sev"])["js_divergence"]
    .mean()
)

mean_jsd_ft, mean_jsd_zs = mean_jsd_ft.align(mean_jsd_zs, join="inner")

#summary stats
print("\nJSD Comparison Across Fine-tuned and Zero-shot Models")
print("Fine-tuned: mean =", mean_jsd_ft.mean(), "std =", mean_jsd_ft.std())
print("Zero-shot:  mean =", mean_jsd_zs.mean(), "std =", mean_jsd_zs.std())

#paired t-test
t_stat_jsd, t_pval_jsd = ttest_rel(mean_jsd_ft, mean_jsd_zs)
print(f"Paired t-test (JSD): t = {t_stat_jsd:.4f}, p = {t_pval_jsd:.4g}")

#wilcoxon signed-rank test
w_stat_jsd, w_pval_jsd = wilcoxon(mean_jsd_ft, mean_jsd_zs)
print(f"Wilcoxon (JSD): W = {w_stat_jsd:.4f}, p = {w_pval_jsd:.4g}")
