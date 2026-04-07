import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import t

# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
base_in = "/Users/bazam/dev/Italian_analysis/data/"
base_out = "/Users/bazam/dev/Italian_analysis/results/"

data_file = os.path.join(base_in, "datacombined-italy2.csv")

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
df = pd.read_csv(data_file)

# Standardize cluster column
df = df.rename(columns={"Cluster": "cluster"})
df["cluster"] = df["cluster"].astype(int)

# -----------------------------------------------------------------------------
# Question groups
# -----------------------------------------------------------------------------
knowledge_items = ["Q1","Q5","Q8","Q11","Q12","Q13","Q14","Q16","Q18","Q22","Q26","Q27"]
attitude_items = ["Q3","Q9","Q10","Q19","Q21","Q29"]
practice_items = ["Q2","Q4","Q20","Q7"]
risk_items = ["Q6","Q15","Q17","Q23","Q24","Q25"]

# -----------------------------------------------------------------------------
# Composite variables (already present but recomputed for safety)
# -----------------------------------------------------------------------------
df["attitude_composite"] = df[attitude_items].mean(axis=1)
df["practice_composite"] = df[practice_items].mean(axis=1)
df["risk_composite"] = df[risk_items].mean(axis=1)

dkapr_vars = [
    "knowledge_score",
    "attitude_composite",
    "practice_composite",
    "risk_composite"
]

# -----------------------------------------------------------------------------
# Normalize values (0–1) for radar plot
# -----------------------------------------------------------------------------
scaler = MinMaxScaler()

normalized = df[["respondent_id","cluster"] + dkapr_vars].copy()
normalized[dkapr_vars] = scaler.fit_transform(normalized[dkapr_vars])

# -----------------------------------------------------------------------------
# Radar plot with individual risk variables
# -----------------------------------------------------------------------------
dkapr_items_vars = [
    "knowledge_score",
    "attitude_composite",
    "practice_composite"
] + risk_items

normalized_items = df[["respondent_id","cluster"] + dkapr_items_vars].copy()

normalized_items[dkapr_items_vars] = scaler.fit_transform(
    normalized_items[dkapr_items_vars]
)

categories = dkapr_items_vars
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(12,12))

for cluster_id, subset in normalized_items.groupby("cluster"):

    values = subset[categories].mean().tolist()
    values += values[:1]

    plt.polar(
        angles,
        values,
        label=f"Cluster {int(cluster_id)}",
        linewidth=2
    )

plt.xticks(angles[:-1], categories, color="grey", size=11)

plt.title(
    "DKAP + Individual Risk Variables by Knowledge-Based Clusters",
    size=16,
    y=1.1
)

plt.legend(loc="upper right", bbox_to_anchor=(1.25,1.1))

plt.tight_layout()

radar_path_items = os.path.join(
    base_out,
    "DKAPR_individual_items_cluster_profiles.png"
)

plt.savefig(radar_path_items, dpi=300, bbox_inches="tight")
plt.close()

print("✅ DKAPR individual-item radar plot saved to:", radar_path_items)

plt.figure(figsize=(12,7))

x = np.arange(len(categories))

for cluster_id, subset in normalized_items.groupby("cluster"):

    data = subset[categories]
    
    means = data.mean()
    sem = data.sem()
    n = len(data)
    
    t_val = t.ppf(0.975, df=n-1) if n > 1 else 0
    ci = sem * t_val

    plt.errorbar(
        x,
        means,
        yerr=ci,
        label=f"Cluster {int(cluster_id)}",
        marker='o',
        capsize=4,
        linewidth=2
    )

plt.xticks(x, categories, rotation=45, ha='right')

plt.ylabel("Normalized Score (0–1)")
plt.title("DKAP + Risk Variables by Cluster (Mean ± 95% CI)")

plt.legend()
plt.tight_layout()

plot_path = os.path.join(base_out, "DKAPR_errorbar_profiles.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print("✅ Error bar plot saved to:", plot_path)

