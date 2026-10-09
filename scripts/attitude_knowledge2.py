
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, bootstrap

# Load dataset
df = pd.read_csv(
    "/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv"
)

# Variables to correlate
pairs = [
    ("knowledge_score", "attitude_score"),
    ("practice_score", "knowledge_score"),
    ("practice_score", "attitude_score"),
    ("risk_score", "knowledge_score"),
    ("risk_score", "attitude_score"),  
    ("risk_score", "practice_score")
]

# Bootstrap 95% CI for Spearman's rho
def spearman_ci(df, var1, var2, n_boot=5000, confidence=0.95):
    sub = df[[var1, var2]].dropna()

    x = sub[var1].to_numpy()
    y = sub[var2].to_numpy()

    if len(sub) < 3:
        return np.nan, np.nan

    def statistic(x, y):
        return spearmanr(x, y).statistic

    result = bootstrap(
        (x, y),
        statistic,
        paired=True,
        n_resamples=n_boot,
        confidence_level=confidence,
        method="percentile",
        rng=np.random.default_rng(42)
    )

    return (
        result.confidence_interval.low,
        result.confidence_interval.high
    )


# Calculate Spearman correlation and CI if p < 0.001
def compute_spearman(df, var1, var2):
    sub = df[[var1, var2]].dropna()
    n = len(sub)

    if n < 3:
        return np.nan, np.nan, n, np.nan, np.nan

    rho, pval = spearmanr(sub[var1], sub[var2])

    ci_low, ci_high = np.nan, np.nan

    if pval < 0.001:
        ci_low, ci_high = spearman_ci(sub, var1, var2)

    return rho, pval, n, ci_low, ci_high


# Store results for the whole sample and each cluster
results = []

groups = [("Whole sample", df)]

for cluster_id, subset in df.groupby("Cluster", dropna=True, sort=True):
    groups.append((f"Cluster {cluster_id}", subset))


for group_name, subset in groups:
    print(f"\nSpearman correlations: {group_name}")

    for var1, var2 in pairs:
        rho, pval, n, ci_low, ci_high = compute_spearman(
            subset, var1, var2
        )

        result = {
            "Group": group_name,
            "Variable_1": var1,
            "Variable_2": var2,
            "Spearman_rho": rho,
            "p_value": pval,
            "N": n,
            "CI_95_lower": ci_low,
            "CI_95_upper": ci_high
        }

        results.append(result)

        if pval < 0.001:
            print(
                f"{var1} vs {var2}: "
                f"rho={rho:.3f}, "
                f"95% CI [{ci_low:.3f}, {ci_high:.3f}], "
                f"p={pval:.6g}, N={n}"
            )
        else:
            print(
                f"{var1} vs {var2}: "
                f"rho={rho:.3f}, "
                f"p={pval:.6g}, N={n}"
            )


# Create and display results table
results_df = pd.DataFrame(results)

results_df = results_df.round({
    "Spearman_rho": 3,
    "p_value": 6,
    "CI_95_lower": 3,
    "CI_95_upper": 3
})

print("\nFull results table:")
print(results_df.to_string(index=False))


# Export results
output_path = ("/Users/bazam/dev/Italian_analysis/results/spearman_correlations_with_CI.csv")

results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")