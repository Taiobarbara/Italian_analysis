
import pandas as pd
import numpy as np
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = "/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv"

SCORE_COLS = [
    "knowledge_score",
    "attitude_score",
    "practice_score",
    "risk_score"
]

CLUSTER_COL = "Cluster"


# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# Check required columns
required_cols = SCORE_COLS + [CLUSTER_COL]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Convert scores to numeric, coercing invalid values to NaN
for col in SCORE_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ── Function to calculate mean and 95% CI ────────────────────────────────────
def mean_ci(data, confidence=0.95):
    """Return n, mean, SD, SE, and t-based confidence interval."""
    x = pd.to_numeric(data, errors="coerce").dropna()
    n = len(x)

    if n == 0:
        return {
            "n": 0,
            "Mean": np.nan,
            "SD": np.nan,
            "SE": np.nan,
            "95% CI lower": np.nan,
            "95% CI upper": np.nan
        }

    mean = x.mean()
    sd = x.std(ddof=1) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan

    if n > 1:
        t_crit = stats.t.ppf(
            (1 + confidence) / 2,
            df=n - 1
        )
        margin = t_crit * se
        lower = mean - margin
        upper = mean + margin
    else:
        lower = np.nan
        upper = np.nan

    return {
        "n": n,
        "Mean": mean,
        "SD": sd,
        "SE": se,
        "95% CI lower": lower,
        "95% CI upper": upper
    }


# ── Calculate results by cluster and for the whole sample ────────────────────
results = []

groups = [("Whole sample", df)] + [
    (f"Cluster {cluster}", group)
    for cluster, group in df.groupby(CLUSTER_COL, dropna=True, sort=True)
]

for group_name, group_df in groups:
    for score in SCORE_COLS:
        stats_dict = mean_ci(group_df[score])

        results.append({
            "Group": group_name,
            "Score": score,
            **stats_dict
        })

results_df = pd.DataFrame(results)

# Round numerical results for presentation
numeric_cols = [
    "Mean", "SD", "SE",
    "95% CI lower", "95% CI upper"
]
results_df[numeric_cols] = results_df[numeric_cols].round(3)

print("\n=== Mean Scores and 95% Confidence Intervals ===")
print(results_df.to_string(index=False))

# Save results
OUTPUT_PATH = (
    "/Users/bazam/dev/Italian_analysis/results/"
    "cluster_score_means_95CI.csv"
)

results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nResults saved to: {OUTPUT_PATH}")