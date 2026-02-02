import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
base_input = "/Users/bazam/dev/Italian_analysis/data/"
base_output = "/Users/bazam/dev/Italian_analysis/results/"

practice_file = os.path.join(base_input, "practice-italy.csv")
knowledge_file = os.path.join(base_input, "demo_clusters.csv")
attitude_file = os.path.join(base_input, "attitude-italy.csv")

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
pract = pd.read_csv(practice_file)
know = pd.read_csv(knowledge_file)
att = pd.read_csv(attitude_file)

# ------------------------------------------------------------
# Compute composite scores
# ------------------------------------------------------------
pract_questions = [col for col in pract.columns if col.startswith("Q")]
pract["practice_composite"] = pract[pract_questions].mean(axis=1, skipna=True)
att["attitude_composite"] = att[[c for c in att.columns if c.startswith("Q")]].mean(axis=1, skipna=True)

# Merge datasets
merged = (
    pract[["respondent_id", "practice_composite"]]
    .merge(know, on="respondent_id", how="inner")
    .merge(att[["respondent_id", "attitude_composite"]], on="respondent_id", how="inner")
)

# ------------------------------------------------------------
# Correlation Analysis
# ------------------------------------------------------------
corr_results = []

for ref, col in [("Knowledge", "knowledge_score"), ("Attitude", "attitude_composite")]:
    rho, p = spearmanr(
        merged["practice_composite"],
        merged[col],
        nan_policy="omit"
    )
    corr_results.append({
        "Reference": ref,
        "Spearman_rho": rho,
        "p_value": p
    })

# ------------------------------------------------------------
# Scatterplot matrix (Knowledge, Attitude, Practice)
# ------------------------------------------------------------
sns.pairplot(
    merged[["knowledge_score", "attitude_composite", "practice_composite"]],
    diag_kind="kde",
    plot_kws={"alpha": 0.6},
)
plt.suptitle("Scatterplot Matrix: Knowledge, Attitude, Practice", y=1.02)
scatter_matrix_path = base_output + "practice_scatter_matrix.png"
plt.savefig(scatter_matrix_path, bbox_inches="tight", dpi=300)
plt.close()

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(base_output + "practice_correlation_results.csv", index=False)

# ------------------------------------------------------------
# Cluster heatmap (mean scores per cluster)
# ------------------------------------------------------------
cluster_means = merged.merge(pract, on="respondent_id")[pract_questions + ["Cluster"]]
cluster_mean_df = cluster_means.groupby("Cluster").mean()

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_mean_df, annot=True, cmap="coolwarm", cbar=True)
plt.title("Mean Practice Scores by Cluster")
heatmap_path = base_output + "practice_cluster_heatmap.png"
plt.savefig(heatmap_path, bbox_inches="tight", dpi=300)
plt.close()