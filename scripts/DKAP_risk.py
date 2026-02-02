import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

# --- Normalization for radar plot (grouped by existing clusters) ---
normalized = df_full.groupby("Cluster")[core_vars[:-1]].mean().reset_index()
normalized[core_vars[:-1]] = normalized[core_vars[:-1]].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

import scikit_posthocs as sp

kw_results = []
dunn_results = []

for var in ["risk_score"] + risk_questions:
    tmp = df_full[["Cluster", var]].dropna()

    if tmp["Cluster"].nunique() > 1:
        # --- Kruskal–Wallis test ---
        grouped = [
            group[var].values
            for _, group in tmp.groupby("Cluster")
        ]

        H, p = stats.kruskal(*grouped)
        kw_results.append((var, H, p))

        # --- Dunn post-hoc test (Bonferroni corrected) ---
        dunn = sp.posthoc_dunn(
            tmp,
            val_col=var,
            group_col="Cluster",
            p_adjust="bonferroni"
        )
        dunn_results.append((var, dunn))

# Save Kruskal-Wallis and Dunn post hoc tests
with open(os.path.join(base_out, "risk_perception_nonparametric_results.txt"), "w") as f:
    f.write("=== Kruskal–Wallis tests ===\n")
    for var, H, p in kw_results:
        f.write(f"\n{var}: H = {H:.3f}, p = {p:.4f}\n")

    f.write("\n\n=== Dunn post-hoc tests (Bonferroni corrected) ===\n")
    for var, table in dunn_results:
        f.write(f"\n{var}:\n")
        f.write(table.to_string())
        f.write("\n")

# --- Boxplots for each risk perception question ---
for var in risk_questions + ["risk_score"]:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_full, x="Cluster", y=var, palette="Set3")
    sns.stripplot(data=df_full, x="Cluster", y=var, color="black", alpha=0.4, jitter=0.15)
    plt.title(f"{var} across Knowledge-Based Clusters")
    plt.xlabel("Knowledge-Based Cluster")
    plt.ylabel(var)
    plt.tight_layout()
    plt.savefig(os.path.join(base_out, f"{var}_boxplot.png"), dpi=300)
    plt.close()

# --- Barplot for composite risk perception (mean per cluster) ---
plt.figure(figsize=(8, 6))
sns.barplot(
    data=df_full,
    x="Cluster",
    y="risk_score",
    estimator=np.median,
    errorbar=("pi", 50),
    palette="pastel"
)
plt.title("Composite Risk Perception across Knowledge-Based Clusters")
plt.xlabel("Cluster")
plt.ylabel("Mean Risk Perception")
plt.tight_layout()
plt.savefig(os.path.join(base_out, "risk_perception_barplot.png"), dpi=300)
plt.close()

# --- Correlation matrix (Risk vs K/A/P) ---
corr_vars = ["knowledge_score", "attitude_composite", "practice_composite", "risk_score"]
corr = df_full[corr_vars].corr(method="spearman")
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix: DKAP + Risk Perception")
plt.tight_layout()
plt.savefig(os.path.join(base_out, "DKAP_Risk_correlation_matrix.png"), dpi=300)
plt.close()

print("\n✅ Risk perception extension completed successfully.")
print(f"Results saved to: {base_out}")
print("- risk_perception_ANOVA_results.txt")
print("- DKAP_Risk_extended_radar.png")
print("- risk_perception_barplot.png")
print("- Boxplots for each risk question (Q6, Q15, Q17, Q26)")