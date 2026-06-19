import pandas as pd
import numpy as np

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/all_answers_binary.csv")

# =========================================================
# VARIABLES OF INTEREST
# =========================================================

questions = {
    "Q6":  [col for col in df.columns if col.startswith("Q6_")],
    "Q15": [col for col in df.columns if col.startswith("Q15_")],
    "Q17": [col for col in df.columns if col.startswith("Q17_")],
    "Q23": [col for col in df.columns if col.startswith("Q23_")],
    "Q24": [col for col in df.columns if col.startswith("Q24_")],
    "Q25": [col for col in df.columns if col.startswith("Q25_")]
}

# =========================================================
# FUNCTION TO COMPUTE DESCRIPTIVE STATISTICS
# =========================================================

def compute_stats(dataframe, variables):
    """
    Computes mean, median and standard deviation
    for binary response variables.
    """
    
    stats = pd.DataFrame({
        "Mean": dataframe[variables].mean(),
        "Median": dataframe[variables].median(),
        "SD": dataframe[variables].std()
    })
    
    return stats.round(3)

# =========================================================
# TOTAL SAMPLE STATISTICS
# =========================================================

total_results = {}

for q, vars_list in questions.items():
    total_results[q] = compute_stats(df, vars_list)

# Print results
print("\n================ TOTAL SAMPLE =================\n")

for q, table in total_results.items():
    print(f"\n----- {q} -----")
    print(table)

# =========================================================
# CLUSTER-LEVEL STATISTICS
# =========================================================

cluster_results = {}

for cluster in sorted(df["Cluster"].unique()):
    
    cluster_df = df[df["Cluster"] == cluster]
    
    cluster_results[cluster] = {}
    
    for q, vars_list in questions.items():
        cluster_results[cluster][q] = compute_stats(cluster_df, vars_list)

# Print cluster results
print("\n================ PER CLUSTER =================\n")

for cluster, q_results in cluster_results.items():
    
    print(f"\n################################################")
    print(f"CLUSTER {cluster}")
    print(f"################################################")
    
    for q, table in q_results.items():
        print(f"\n----- {q} -----")
        print(table)

# =========================================================
# OPTIONAL: EXPORT RESULTS TO EXCEL
# =========================================================

with pd.ExcelWriter("descriptive_statistics_by_cluster.xlsx") as writer:
    
    # Total sample sheets
    for q, table in total_results.items():
        table.to_excel(writer, sheet_name=f"Total_{q}")
    
    # Cluster sheets
    for cluster, q_results in cluster_results.items():
        for q, table in q_results.items():
            sheet_name = f"C{cluster}_{q}"
            table.to_excel(writer, sheet_name=sheet_name[:31])  # Excel limit

print("\nResults exported to:")
print("descriptive_statistics_by_cluster.xlsx")