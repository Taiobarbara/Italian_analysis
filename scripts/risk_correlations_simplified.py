import os
import pandas as pd
from scipy.stats import spearmanr

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
# Variables
# ------------------------------------------------------------
predictors = [
    "knowledge_score",
    "attitude_score",
    "practice_score"
]

outcomes = [
    "risk_score",
    "Q6",
    "Q15",
    "Q17",
    "Q23",
    "Q24",
    "Q25"
]

# ------------------------------------------------------------
# Function to calculate correlations
# ------------------------------------------------------------
def calculate_correlations(data, label):

    results = []

    for outcome in outcomes:

        row = {
            "Group": label,
            "Variable": outcome
        }

        for predictor in predictors:

            # Pairwise complete observations
            temp = data[[predictor, outcome]].dropna()

            # Spearman correlation
            rho, pval = spearmanr(
                temp[predictor],
                temp[outcome]
            )

            prefix = predictor.replace("_score", "").capitalize()

            # Store results
            row[f"{prefix} rho"] = round(rho, 4)

            # Scientific notation prevents tiny p-values from becoming 0
            row[f"{prefix} p"] = f"{pval:.10e}"

        results.append(row)

    return results

# ------------------------------------------------------------
# Build full results table
# ------------------------------------------------------------
all_results = []

# Whole sample
all_results.extend(
    calculate_correlations(df, "Whole sample")
)

# Individual clusters
for cluster in sorted(df["cluster"].unique()):

    cluster_df = df[df["cluster"] == cluster]

    all_results.extend(
        calculate_correlations(
            cluster_df,
            f"Cluster {cluster}"
        )
    )

# ------------------------------------------------------------
# Convert to dataframe
# ------------------------------------------------------------
corr_table = pd.DataFrame(all_results)

# ------------------------------------------------------------
# Print table
# ------------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print("\nSpearman Correlation Results\n")
print(corr_table.to_string(index=False))

# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------
output_file = os.path.join(
    base_out,
    "DKAP_vs_risk_correlations_by_cluster.csv"
)

corr_table.to_csv(
    output_file,
    index=False
)

print(f"\n✅ Correlation table saved:\n{output_file}")