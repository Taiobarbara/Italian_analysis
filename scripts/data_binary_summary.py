import pandas as pd

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
data_file = "/Users/bazam/dev/Italian_analysis/data/all_answers_binary.csv"

df = pd.read_csv(data_file)

# ------------------------------------------------------------
# Variables to summarize
# ------------------------------------------------------------
# Exclude Cluster + respondent_id
variables = [
    col for col in df.columns
    if col not in ["Cluster", "respondent_id"]
]

# ------------------------------------------------------------
# Function to build summary rows
# ------------------------------------------------------------
def build_summary(data, label):

    n = len(data)

    count_row = {
        "Cluster": label,
        "Metric": "count"
    }

    percent_row = {
        "Cluster": "",
        "Metric": "percentage"
    }

    for var in variables:

        # Count respondents with value == 1
        count = (data[var] == 1).sum()

        percentage = (count / n) * 100 if n > 0 else 0

        count_row[var] = count
        percent_row[var] = f"{percentage:.0f}%"

    return [count_row, percent_row]

# ------------------------------------------------------------
# Build summary table
# ------------------------------------------------------------
rows = []

# Whole sample
rows.extend(build_summary(df, "Whole sample"))

# Cluster summaries
for cluster in sorted(df["Cluster"].unique()):

    cluster_df = df[df["Cluster"] == cluster]

    rows.extend(
        build_summary(cluster_df, f"Cluster {cluster}")
    )

# ------------------------------------------------------------
# Convert to dataframe
# ------------------------------------------------------------
summary_table = pd.DataFrame(rows)

# ------------------------------------------------------------
# Print table
# ------------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

print("\nBinary Response Summary Table\n")
print(summary_table.to_string(index=False))

# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------
output_file = "/Users/bazam/dev/Italian_analysis/results/binary_response_summary_table.csv"

summary_table.to_csv(output_file, index=False)

print(f"\n✅ Table saved to:\n{output_file}")