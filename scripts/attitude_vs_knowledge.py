import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kruskal
import scikit_posthocs as sp

# === File paths ===
base_input = "/Users/bazam/dev/Italian_analysis/data/"
base_output = "/Users/bazam/dev/Italian_analysis/results/"

attitude_file = os.path.join(base_input, "attitude-italy.csv")
knowledge_file = os.path.join(base_input, "demo_clusters.csv")

output_corr = os.path.join(base_output, "attitude_question_spearman_correlations.csv")
output_kw = os.path.join(base_output, "attitude_kruskal_results.csv")
output_dunn = os.path.join(base_output, "attitude_dunn_posthoc.csv")

# === Load data ===
df_aw = pd.read_csv(attitude_file)
df_know = pd.read_csv(knowledge_file)

# Merge on respondent_id
df = df_aw.merge(df_know, on="respondent_id", how="left")
df.rename(columns={"cluster": "cluster_label"}, inplace=True)

print(f"✅ Data merged successfully: {df.shape[0]} respondents")

# Identify awareness question columns
attitude_cols = [c for c in df.columns if c.startswith("Q")]

# =============================================================================
# 1. Spearman correlations with knowledge score
# =============================================================================
corrs = []

for q in attitude_cols:
    valid = df[[q, "knowledge_score"]].dropna()
    if len(valid) > 5:
        rho, p = spearmanr(valid[q], valid["knowledge_score"])
        corrs.append({
            "Question": q,
            "Spearman_rho": rho,
            "p_value": p
        })

corr_df = pd.DataFrame(corrs)
corr_df.to_csv(output_corr, index=False)

print(f"📈 Spearman correlations saved to: {output_corr}")
print(corr_df.round(3))

# =============================================================================
# 2. Attitude differences across knowledge clusters (Kruskal–Wallis)
# =============================================================================
kw_results = []

for q in attitude_cols:
    groups = [
        g[q].dropna()
        for _, g in df.groupby("Cluster")
        if len(g[q].dropna()) > 0
    ]
    if len(groups) > 1:
        H, p = kruskal(*groups)
        kw_results.append({
            "Question": q,
            "H_statistic": H,
            "p_value": p
        })

kw_df = pd.DataFrame(kw_results)
kw_df.to_csv(output_kw, index=False)

print(f"📊 Kruskal–Wallis results saved to: {output_kw}")
print(kw_df.round(3))

# =============================================================================
# 3. Post-hoc Dunn tests (Bonferroni) for significant results
# =============================================================================
dunn_results = []

for q in kw_df.loc[kw_df["p_value"] < 0.05, "Question"]:
    sub = df[["Cluster", q]].dropna()
    dunn = sp.posthoc_dunn(
        sub,
        val_col=q,
        group_col="Cluster",
        p_adjust="bonferroni"
    )

    dunn_long = (
        dunn.reset_index()
        .melt(id_vars="index", var_name="cluster_2", value_name="p_adj")
        .rename(columns={"index": "cluster_1"})
    )
    dunn_long["Question"] = q
    dunn_results.append(dunn_long)

if dunn_results:
    dunn_df = pd.concat(dunn_results, ignore_index=True)
    dunn_df.to_csv(output_dunn, index=False)
    print(f"📘 Dunn post-hoc results saved to: {output_dunn}")

print("✅ Non-parametric attitude analysis completed successfully.")