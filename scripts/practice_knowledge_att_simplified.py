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
# 1b. Bootstrap 95% CIs for selected Spearman correlations
# Q2 and Q4 vs Knowledge and Attitude scores
# =============================================================================
from scipy.stats import bootstrap

for q in ["Q2", "Q4"]:
    for score in ["knowledge_score", "attitude_score"]:

        valid = df[[q, score]].dropna()
        x = valid[q].to_numpy()
        y = valid[score].to_numpy()

        if len(valid) < 6:
            print(f"{q} vs {score}: insufficient data (N={len(valid)})")
            continue

        rho, p = spearmanr(x, y)

        def spearman_statistic(x, y):
            return spearmanr(x, y).statistic

        boot_result = bootstrap(
            (x, y),
            spearman_statistic,
            paired=True,
            n_resamples=5000,
            confidence_level=0.95,
            method="percentile",
            rng=np.random.default_rng(42)
        )

        ci_low = boot_result.confidence_interval.low
        ci_high = boot_result.confidence_interval.high

        print(
            f"{q} vs {score}: "
            f"rho = {rho:.3f}, "
            f"95% CI [{ci_low:.3f}, {ci_high:.3f}], "
            f"p = {p:.6g}, "
            f"N = {len(valid)}"
        )

