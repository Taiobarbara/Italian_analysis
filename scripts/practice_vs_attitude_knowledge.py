import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr, kruskal
import scikit_posthocs as sp

# =============================================================================
# 📁 Paths
# =============================================================================
base_input = "/Users/bazam/dev/Italian_analysis/data/"
base_output = "/Users/bazam/dev/Italian_analysis/results/"

data_file = os.path.join(base_input, "datacombined-italy2.csv")

# =============================================================================
# 📖 Load data
# =============================================================================
df = pd.read_csv(data_file)

# Ensure cluster column is lowercase and numeric
df = df.rename(columns={"Cluster": "cluster"})
df["cluster"] = df["cluster"].astype(int)

print(df.columns.tolist())

# =============================================================================
# 🔢 Variable definitions
# =============================================================================

knowledge_items = ["Q1","Q5","Q8","Q11","Q12","Q13","Q14","Q16","Q18","Q22","Q26","Q27"]

attitude_items = ["Q3","Q9","Q10","Q19","Q21","Q29"]

practice_items = ["Q2","Q4","Q20","Q7"]

risk_items = ["Q6","Q15","Q17","Q23","Q24","Q25"]

# Composite means (if needed)
df["attitude_mean"] = df[attitude_items].mean(axis=1)
df["practice_mean"] = df[practice_items].mean(axis=1)

# =============================================================================
# 1️⃣ Spearman correlations
# Practice vs Knowledge and Attitude
# =============================================================================
corr_results = []

for q in practice_items:

    valid_k = df[[q, "knowledge_score"]].dropna()
    valid_a = df[[q, "attitude_score"]].dropna()

    if len(valid_k) > 5:
        rho_k, p_k = spearmanr(valid_k[q], valid_k["knowledge_score"])
        corr_results.append(["Knowledge", q, rho_k, p_k])

    if len(valid_a) > 5:
        rho_a, p_a = spearmanr(valid_a[q], valid_a["attitude_score"])
        corr_results.append(["Attitude", q, rho_a, p_a])

corr_df = pd.DataFrame(
    corr_results,
    columns=["Reference", "Question", "Spearman_rho", "p_value"]
)

corr_df.to_csv(
    os.path.join(base_output, "practice_spearman_correlations.csv"),
    index=False
)

print("Spearman correlations saved")

# =============================================================================
# 2️⃣ Kruskal–Wallis across knowledge clusters
# =============================================================================
kw_results = []

for q in practice_items:

    groups = [
        g[q].dropna()
        for _, g in df.groupby("cluster")
        if g[q].notna().sum() > 3
    ]

    if len(groups) > 1:
        H, p = kruskal(*groups)
        kw_results.append([q, H, p])

kw_df = pd.DataFrame(
    kw_results,
    columns=["Question", "H_statistic", "p_value"]
)

kw_df.to_csv(
    os.path.join(base_output, "practice_kruskal_results.csv"),
    index=False
)

print("Kruskal–Wallis results saved")

# =============================================================================
# 3️⃣ Dunn post-hoc tests
# =============================================================================
for _, row in kw_df.iterrows():

    if row["p_value"] < 0.05:

        q = row["Question"]

        dunn = sp.posthoc_dunn(
            df,
            val_col=q,
            group_col="cluster",
            p_adjust="holm"
        )

        dunn_long = (
            dunn.reset_index()
            .melt(id_vars="index", var_name="cluster_2", value_name="p_adj")
            .rename(columns={"index": "cluster_1"})
        )

        dunn_long["Question"] = q

        out_path = os.path.join(base_output, f"practice_{q}_dunn_posthoc.csv")
        dunn_long.to_csv(out_path, index=False)

        print(f"Dunn post-hoc saved for {q}")

# =============================================================================
# 4️⃣ Visualisation: Boxplots by cluster
# =============================================================================
sns.set(style="whitegrid")

for q in practice_items:

    plt.figure(figsize=(8,5))

    sns.boxplot(
        data=df,
        x="cluster",
        y=q,
        showfliers=False
    )

    sns.stripplot(
        data=df,
        x="cluster",
        y=q,
        color="black",
        alpha=0.25,
        jitter=0.25,
        size=3
    )

    plt.title(f"{q}: Practice Distribution by Knowledge Cluster")
    plt.xlabel("Knowledge-Based Cluster")
    plt.ylabel("Practice Score")

    plt.tight_layout()

    out_path = os.path.join(base_output, f"practice_{q}_by_cluster.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

# =============================================================================
# 5️⃣ Visualisation: Practice composite by cluster
# =============================================================================
plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="cluster",
    y="practice_score",
    showfliers=False
)

sns.stripplot(
    data=df,
    x="cluster",
    y="practice_score",
    color="black",
    alpha=0.4,
    jitter=0.15,
    size=3
)

plt.title("Practice Composite Score by Knowledge-Based Cluster")
plt.xlabel("Knowledge-Based Clusters")
plt.ylabel("Practice Score")

plt.tight_layout()

out_path = os.path.join(base_output, "practice_composite_by_cluster.png")
plt.savefig(out_path, dpi=300)
plt.close()

print("Practice vs knowledge analysis complete.")