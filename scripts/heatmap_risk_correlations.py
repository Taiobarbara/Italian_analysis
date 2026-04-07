import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
base_in = "/Users/bazam/dev/Italian_analysis/data/"
base_out = "/Users/bazam/dev/Italian_analysis/results/"

data_file = os.path.join(base_in, "datacombined-italy2.csv")

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
df = pd.read_csv(data_file)

# Standardize cluster column
df = df.rename(columns={"Cluster": "cluster"})
df["cluster"] = df["cluster"].astype(int)

# ------------------------------------------------------------
# Question groups
# ------------------------------------------------------------
knowledge_items = ["Q1","Q5","Q8","Q11","Q12","Q13","Q14","Q16","Q18","Q22","Q26","Q27"]
attitude_items = ["Q3","Q9","Q10","Q19","Q21","Q29"]
practice_items = ["Q2","Q4","Q20","Q7"]
risk_items = ["Q6","Q15","Q17","Q23","Q24","Q25"]

# ------------------------------------------------------------
# ORIGINAL: Correlation matrix (Spearman, composites)
# ------------------------------------------------------------
corr_vars = [
    "knowledge_score",
    "attitude_score",
    "practice_score",
    "risk_score"
]

corr = df[corr_vars].corr(method="spearman")

plt.figure(figsize=(6,5))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Matrix: DKAP + MP Risk")

plt.tight_layout()

plt.savefig(
    os.path.join(base_out, "DKAP_MPRisk_correlation_matrix.png"),
    dpi=300
)

plt.close()

print("✅ DKAP + risk subset correlation matrix saved.")

# ------------------------------------------------------------
# NEW: Correlation matrix with individual risk items
# ------------------------------------------------------------
corr_vars_items = [
    "knowledge_score",
    "attitude_score",
    "practice_score"
] + risk_items

corr_items = df[corr_vars_items].corr(method="spearman")

plt.figure(figsize=(10,8))

sns.heatmap(
    corr_items,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Matrix: DKAP vs Individual Risk Items")

plt.tight_layout()

plt.savefig(
    os.path.join(base_out, "DKAP_vs_risk_items_correlation_matrix.png"),
    dpi=300
)

plt.close()

print("✅ Full correlation matrix with individual risk items saved.")

# ------------------------------------------------------------
# NEW: Focused matrix (DKAP vs risk items only)
# ------------------------------------------------------------
corr_subset = corr_items.loc[
    ["knowledge_score","attitude_score","practice_score"],
    risk_items
]

plt.figure(figsize=(10,4))

sns.heatmap(
    corr_subset,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("DKAP vs Risk Items (Spearman Correlation)")

plt.tight_layout()

plt.savefig(
    os.path.join(base_out, "DKAP_vs_risk_items_subset.png"),
    dpi=300
)

plt.close()

print("✅ Focused DKAP vs risk items correlation matrix saved.")