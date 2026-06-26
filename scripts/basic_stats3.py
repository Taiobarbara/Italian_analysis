import pandas as pd
import numpy as np

# =========================================================
# LOAD DATA
# =========================================================

input_file = "/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv"

df = pd.read_csv(input_file)

# =========================================================
# SETTINGS
# =========================================================

score_col = "risk_score"

cluster_col = "Cluster"

# =========================================================
# SCORE INTERVALS
# =========================================================

bins = [0, 0.25, 0.5, 0.75, 1.000001]

labels = [
    "0-0.25",
    "0.25-0.5",
    "0.5-0.75",
    "0.75-1"
]

# =========================================================
# FUNCTION: INTERVAL DISTRIBUTION
# =========================================================

def calculate_interval_distribution(data, score_col):

    categorized = pd.cut(
        data[score_col],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    percentages = (
        categorized.value_counts(normalize=True)
        .reindex(labels, fill_value=0)
        * 100
    )

    return percentages

# =========================================================
# FUNCTION: DESCRIPTIVE STATISTICS
# =========================================================

def calculate_descriptive_stats(data, score_col):

    return {
        "Mean": data[score_col].mean(),
        "Median": data[score_col].median(),
        "SD": data[score_col].std()
    }

# =========================================================
# TOTAL SAMPLE
# =========================================================

interval_results = []
stats_results = []

# -------------------------
# Total Sample
# -------------------------

total_intervals = calculate_interval_distribution(
    df,
    score_col
)

total_stats = calculate_descriptive_stats(
    df,
    score_col
)

interval_row = {
    "Group": "Total Sample"
}

for label in labels:
    interval_row[label] = total_intervals[label]

interval_results.append(interval_row)

stats_row = {
    "Group": "Total Sample",
    **total_stats
}

stats_results.append(stats_row)

# =========================================================
# PER CLUSTER
# =========================================================

clusters = sorted(df[cluster_col].dropna().unique())

for cluster in clusters:

    cluster_df = df[df[cluster_col] == cluster]

    # -------------------------
    # Intervals
    # -------------------------

    cluster_intervals = calculate_interval_distribution(
        cluster_df,
        score_col
    )

    interval_row = {
        "Group": f"Cluster {cluster}"
    }

    for label in labels:
        interval_row[label] = cluster_intervals[label]

    interval_results.append(interval_row)

    # -------------------------
    # Descriptive stats
    # -------------------------

    cluster_stats = calculate_descriptive_stats(
        cluster_df,
        score_col
    )

    stats_row = {
        "Group": f"Cluster {cluster}",
        **cluster_stats
    }

    stats_results.append(stats_row)

# =========================================================
# CREATE TABLES
# =========================================================

interval_table = pd.DataFrame(interval_results)

stats_table = pd.DataFrame(stats_results)

# =========================================================
# ROUND RESULTS
# =========================================================

for table in [interval_table, stats_table]:

    numeric_cols = table.select_dtypes(include=np.number).columns

    table[numeric_cols] = table[numeric_cols].round(4)

# =========================================================
# EXPORT TO EXCEL
# =========================================================

output_file = "risk_score_cluster_analysis.xlsx"

with pd.ExcelWriter(output_file) as writer:

    interval_table.to_excel(
        writer,
        sheet_name="Intervals",
        index=False
    )

    stats_table.to_excel(
        writer,
        sheet_name="Descriptive_Stats",
        index=False
    )

print("Analysis complete.")
print(f"Results saved to: {output_file}")