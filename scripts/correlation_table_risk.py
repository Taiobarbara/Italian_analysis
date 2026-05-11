import os
import pandas as pd
from scipy.stats import spearmanr

# ------------------------------------------------------------
# Path
# ------------------------------------------------------------
data_file = "/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv"

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
# Build correlation summary table
# ------------------------------------------------------------
results = []

for outcome in outcomes:
    row = {"Variable": outcome}

    for predictor in predictors:

        # Pairwise complete observations
        temp = df[[predictor, outcome]].dropna()

        # Spearman correlation
        rho, pval = spearmanr(
            temp[predictor],
            temp[outcome]
        )

        prefix = predictor.replace("_score", "").capitalize()

        # Store values
        row[f"{prefix} rho"] = round(rho, 4)

        # Scientific notation prevents tiny p-values becoming 0
        row[f"{prefix} p"] = f"{pval:.10e}"

    results.append(row)

# ------------------------------------------------------------
# Create dataframe
# ------------------------------------------------------------
corr_table = pd.DataFrame(results)

# ------------------------------------------------------------
# Print table
# ------------------------------------------------------------
print("\nSpearman Correlation Results\n")
print(corr_table.to_string(index=False))