import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
base_in = "/Users/bazam/dev/Italian_analysis/data/"
base_out = "/Users/bazam/dev/Italian_analysis/results/"

data_file = os.path.join(base_in, "datacombined-italy2.csv")

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv(data_file)

# Standardise cluster column
df = df.rename(columns={"Cluster": "cluster"})
df["cluster"] = df["cluster"].astype(int)

# ------------------------------------------------------------
# Question definitions
# ------------------------------------------------------------
knowledge_items = ["Q1","Q5","Q8","Q11","Q12","Q13","Q14","Q16","Q18","Q22","Q26","Q27"]
attitude_items = ["Q3","Q9","Q10","Q19","Q21","Q29"]
practice_items = ["Q2","Q4","Q20","Q7"]
risk_items = ["Q6","Q15","Q17","Q23","Q24","Q25"]

# ------------------------------------------------------------
# Optional: recompute risk score if needed
# ------------------------------------------------------------
df["risk_score"] = df[risk_items].mean(axis=1)

# ------------------------------------------------------------
# Radar plot preparation (cluster means)
# ------------------------------------------------------------
core_vars = ["knowledge_score","attitude_score","practice_score","risk_score"]

cluster_means = df.groupby("cluster")[core_vars].mean().reset_index()

# Normalize for radar
norm = cluster_means.copy()
norm[core_vars] = norm[core_vars].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

# ------------------------------------------------------------
# Kruskal–Wallis + Dunn tests
# ------------------------------------------------------------
kw_results = []
dunn_results = []

for var in ["risk_score"] + risk_items:

    tmp = df[["cluster",var]].dropna()

    if tmp["cluster"].nunique() > 1:

        groups = [
            g[var].values
            for _, g in tmp.groupby("cluster")
        ]

        H, p = stats.kruskal(*groups)
        kw_results.append((var,H,p))

        dunn = sp.posthoc_dunn(
            tmp,
            val_col=var,
            group_col="cluster",
            p_adjust="bonferroni"
        )

        dunn_results.append((var,dunn))

# Save results
with open(os.path.join(base_out,"risk_perception_nonparametric_results.txt"),"w") as f:

    f.write("=== Kruskal–Wallis tests ===\n")

    for var,H,p in kw_results:
        f.write(f"\n{var}: H={H:.3f}, p={p:.4f}\n")

    f.write("\n\n=== Dunn post-hoc tests ===\n")

    for var,table in dunn_results:
        f.write(f"\n{var}\n")
        f.write(table.to_string())
        f.write("\n")

# ------------------------------------------------------------
# Boxplots per risk question
# ------------------------------------------------------------
for var in risk_items + ["risk_score"]:

    plt.figure(figsize=(8,6))

    sns.boxplot(
        data=df,
        x="cluster",
        y=var,
        palette="Set3"
    )

    sns.stripplot(
        data=df,
        x="cluster",
        y=var,
        color="black",
        alpha=0.4,
        jitter=0.15
    )

    plt.title(f"{var} across Knowledge-Based Clusters")
    plt.xlabel("Knowledge-Based Cluster")
    plt.ylabel("Risk perception")

    plt.tight_layout()

    plt.savefig(
        os.path.join(base_out,f"{var}_boxplot.png"),
        dpi=300
    )

    plt.close()

# ------------------------------------------------------------
# Risk question distribution
# ------------------------------------------------------------
risk_long = df.melt(
    id_vars=["respondent_id","cluster"],
    value_vars=risk_items,
    var_name="Risk_Question",
    value_name="Risk_Value"
).dropna()

plt.figure(figsize=(8,6))

sns.boxplot(
    data=risk_long,
    x="Risk_Question",
    y="Risk_Value",
    showfliers=False
)

sns.stripplot(
    data=risk_long,
    x="Risk_Question",
    y="Risk_Value",
    color="black",
    alpha=0.3,
    jitter=0.2,
    size=3
)

plt.title("Risk Perception Distribution per Question")
plt.xlabel("Question")
plt.ylabel("Risk perception")

plt.tight_layout()

plt.savefig(
    os.path.join(base_out,"risk_perception_Qs_boxplot.png"),
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# Correlation matrix (DKAP + Risk)
# ------------------------------------------------------------
corr_vars = ["knowledge_score","attitude_score","practice_score","risk_score"]

corr = df[corr_vars].corr(method="spearman")

plt.figure(figsize=(6,5))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Matrix: DKAP + Risk Perception")

plt.tight_layout()

plt.savefig(
    os.path.join(base_out,"DKAP_Risk_correlation_matrix.png"),
    dpi=300
)

plt.close()

print("\n✅ Risk perception analysis completed successfully.")
print(f"Results saved to: {base_out}")