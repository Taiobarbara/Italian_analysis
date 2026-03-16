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
# Recompute composites (optional safety step)
# ------------------------------------------------------------
df["attitude_composite"] = df[attitude_items].mean(axis=1)
df["practice_composite"] = df[practice_items].mean(axis=1)

# ------------------------------------------------------------
# Risk subset mean (example subset of risk items)
# ------------------------------------------------------------
risk_subset = ["Q15","Q17","Q23"]

df["risk_subset_mean"] = df[risk_subset].mean(axis=1)

# Check missingness
print(df[risk_subset + ["risk_subset_mean"]].isna().mean())

# ------------------------------------------------------------
# Correlation matrix (Spearman)
# ------------------------------------------------------------
corr_vars = [
    "knowledge_score",
    "attitude_composite",
    "practice_composite",
    "risk_subset_mean"
]

corr = df[corr_vars].corr(method="spearman")

# ------------------------------------------------------------
# Heatmap
# ------------------------------------------------------------
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