import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr

# Paths
base_input = "/Users/bazam/dev/Italian_analysis/data/"
base_output = "/Users/bazam/dev/Italian_analysis/results/"

data_file = os.path.join(base_input, "datacombined-italy2.csv")

# Load data
df = pd.read_csv(data_file)

# Standardise cluster column
df = df.rename(columns={"Cluster": "cluster"})
df["cluster"] = df["cluster"].astype(int)

# Question definitions
practice_items = ["Q2","Q4","Q7","Q20"]

df["practice_composite"] = df["practice_score"]

# Correlation Analysis

corr_results = []

for ref, col in [
    ("Knowledge", "knowledge_score"),
    ("Attitude", "attitude_score")
]:

    rho, p = spearmanr(
        df["practice_composite"],
        df[col],
        nan_policy="omit"
    )

    corr_results.append({
        "Reference": ref,
        "Spearman_rho": rho,
        "p_value": p
    })

corr_df = pd.DataFrame(corr_results)

corr_df.to_csv(
    os.path.join(base_output, "practice_correlation_results.csv"),
    index=False
)

print("Correlation results saved")

# Scatterplot matrix (Knowledge, Attitude, Practice)
sns.pairplot(
    df[["knowledge_score","attitude_score","practice_composite"]],
    diag_kind="kde",
    plot_kws={"alpha":0.6}
)

plt.suptitle(
    "Scatterplot Matrix: Knowledge, Attitude, Practice",
    y=1.02
)

scatter_matrix_path = os.path.join(
    base_output,
    "practice_scatter_matrix.png"
)

plt.savefig(scatter_matrix_path, bbox_inches="tight", dpi=300)
plt.close()

print("Scatter matrix saved")

# Cluster heatmap (mean practice scores per cluster)
cluster_mean_df = (
    df.groupby("cluster")[practice_items]
    .mean()
)

plt.figure(figsize=(8,5))

sns.heatmap(
    cluster_mean_df,
    annot=True,
    cmap="coolwarm",
    cbar=True
)

plt.title("Mean Practice Scores by Knowledge Cluster")
plt.xlabel("Practice Question")
plt.ylabel("Cluster")

heatmap_path = os.path.join(
    base_output,
    "practice_cluster_heatmap.png"
)

plt.savefig(heatmap_path, bbox_inches="tight", dpi=300)
plt.close()

print("Cluster heatmap saved")