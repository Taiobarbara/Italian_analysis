import pandas as pd

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
data_file = "/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv"

df = pd.read_csv(data_file)

# ------------------------------------------------------------
# Variables to summarize
# ------------------------------------------------------------
variables = [
    "Q1","Q5","Q8","Q11","Q12","Q13","Q14","Q16","Q18","Q22","Q26","Q27",
    "knowledge_score",
    "Q3","Q9","Q10","Q19","Q21","Q29",
    "attitude_score",
    "Q2","Q4","Q7","Q20",
    "practice_score",
    "Q6","Q15","Q17","Q23","Q24","Q25",
    "risk_score",
    "gender_female","gender_male","gender_non-binary","gender_prefer_not_to_say",
    "age_less_than_18","age_19-30","age_31-40","age_41-55","age_56-65","age_over_65",
    "educational_level_primary_school",
    "educational_level_high_school",
    "educational_level_bachelors_degree",
    "educational_level_masters_degree",
    "educational_level_PhD_or_equivalent"
]

# ------------------------------------------------------------
# Function to build rows
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

        # ----------------------------------------------------
        # Binary/categorical variables
        # Count values equal to 1 or 2
        # ----------------------------------------------------
        if (
            var.startswith("Q")
            or var.startswith("gender_")
            or var.startswith("age_")
            or var.startswith("educational_level_")
        ):

            # For survey items:
            # assumes 2 = positive/correct
            if var.startswith("Q"):
                count = (data[var] == 2).sum()

            # For dummy-coded demographic variables:
            # assumes 1 = belongs to category
            else:
                count = (data[var] == 1).sum()

            percentage = (count / n) * 100 if n > 0 else 0

            count_row[var] = count
            percent_row[var] = f"{percentage:.0f}%"

        # ----------------------------------------------------
        # Continuous/composite scores
        # Show mean instead
        # ----------------------------------------------------
        else:

            mean_value = data[var].mean()

            count_row[var] = round(mean_value, 3)
            percent_row[var] = ""

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
# Create dataframe
# ------------------------------------------------------------
summary_table = pd.DataFrame(rows)

# ------------------------------------------------------------
# Print full table
# ------------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print("\nCluster Summary Table\n")
print(summary_table.to_string(index=False))

# ------------------------------------------------------------
# Save table as CSV
# ------------------------------------------------------------
output_file = "/Users/bazam/dev/Italian_analysis/results/cluster_summary_table.csv"

summary_table.to_csv(output_file, index=False)

print(f"\n✅ Table saved to:\n{output_file}")