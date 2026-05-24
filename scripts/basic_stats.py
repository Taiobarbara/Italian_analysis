import pandas as pd
import numpy as np

# =========================================================
# LOAD DATA
# =========================================================

# Replace with your file path
input_file = "your_dataset.csv"

df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/all_answers_binary.csv")

# =========================================================
# IDENTIFY VARIABLES
# =========================================================

# Columns to exclude from analysis
exclude_cols = ["respondent_id", "Cluster"]

# All questionnaire variables
question_cols = [col for col in df.columns if col not in exclude_cols]

# Ensure numeric
df[question_cols] = df[question_cols].apply(pd.to_numeric, errors='coerce')

# =========================================================
# FUNCTION TO CALCULATE STATS
# =========================================================

def calculate_stats(data, cols):
    """
    Returns dataframe with Mean, Median, SD
    """
    
    stats = pd.DataFrame(index=cols)

    # Mean as percentage
    stats["Mean"] = data[cols].mean() * 100

    # Median as percentage
    stats["Median"] = data[cols].median() * 100

    # Standard deviation as percentage points
    stats["SD"] = data[cols].std() * 100

    return stats

# =========================================================
# TOTAL SAMPLE STATS
# =========================================================

results = {}

results["total sample"] = calculate_stats(df, question_cols)

# =========================================================
# PER-CLUSTER STATS
# =========================================================

clusters = sorted(df["Cluster"].dropna().unique())

for cluster in clusters:
    
    cluster_df = df[df["Cluster"] == cluster]
    
    results[f"Cluster {cluster}"] = calculate_stats(cluster_df, question_cols)

# =========================================================
# COMBINE RESULTS
# =========================================================

final_table = pd.concat(results, axis=1)

# Optional: round values
final_table = final_table.round(2)

# =========================================================
# EXPORT CSV
# =========================================================

output_file = "cluster_statistics.csv"

final_table.to_csv(output_file)

print("Analysis complete.")
print(f"Output saved to: {output_file}")