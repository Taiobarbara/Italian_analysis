import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
from scipy import stats


# --- Paths ---
base_in = "/Users/bazam/dev/Italian_analysis/data/"
base_out = "/Users/bazam/dev/Italian_analysis/results/"

practice_file = os.path.join(base_in, "practice-italy.csv")
knowledge_file = os.path.join(base_in, "demo_clusters.csv")
attitude_file = os.path.join(base_in, "attitude-italy.csv")
risk_path = os.path.join(base_in, "risk-italy.csv")

# --- Load data ---
df_k = pd.read_csv(knowledge_file)
df_a = pd.read_csv(attitude_file)
df_p = pd.read_csv(practice_file)
df_r = pd.read_csv(risk_path)

# --- Clean risk data ---
df_r = df_r.replace(0, np.nan)
risk_questions = [c for c in df_r.columns if c != 'respondent_id']
df_r['risk_score'] = df_r[risk_questions].mean(axis=1)


# --- Merge datasets (keeping knowledge clusters fixed) ---
df_full = (
    df_k
    .merge(df_a, on="respondent_id", how="left", suffixes=("", "_attitude"))
    .merge(df_p, on="respondent_id", how="left", suffixes=("", "_practice"))
    .merge(df_r, on="respondent_id", how="left", suffixes=("", "_risk"))
)
# Identify risk question columns
risk_cols = ["Q15", "Q17", "Q26"]

# --- Compute mean risk score from selected risk items ---
df_full["risk_subset_mean"] = df_full[risk_cols].mean(axis=1)

# Confirm that cluster variable is from the knowledge file
assert "Cluster" in df_k.columns, "⚠️ The 'cluster' variable must come from the knowledge dataset."
print(f"Cluster variable sourced from knowledge file with {df_full['Cluster'].nunique()} unique clusters.")

# --- Compute composite attitude and practice if not present ---
if "attitude_composite" not in df_full.columns:
    attitude_cols = [c for c in df_a.columns if c.startswith("Q")]
    df_full["attitude_composite"] = df_full[attitude_cols].mean(axis=1)

if "practice_composite" not in df_full.columns:
    practice_cols = [c for c in df_p.columns if c.startswith("Q")]
    df_full["practice_composite"] = df_full[practice_cols].mean(axis=1)

# --- Select DKAP + Risk variables ---
core_vars = ["knowledge_score", "attitude_composite", "practice_composite", "risk_score", "Cluster"] 

print(df_full[risk_cols + ["risk_subset_mean"]].isna().mean())

# --- Correlation matrix (Risk subset vs K/A/P) ---
corr_vars = [
    "knowledge_score",
    "attitude_composite",
    "practice_composite",
    "risk_subset_mean"
]

corr = df_full[corr_vars].corr(method="spearman")

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix: DKAP + MP Risk")
plt.tight_layout()
plt.savefig(
    os.path.join(base_out, "DKAP_MPRisk_correlation_matrix.png"),
    dpi=300
)
plt.close()
