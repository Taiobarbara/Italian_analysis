import pandas as pd
import numpy as np

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv")

# =========================================================
# QUESTIONS USED TO BUILD RISK SCORE
# =========================================================

risk_questions = ["Q6", "Q15", "Q17", "Q23", "Q24", "Q25"]

# =========================================================
# FUNCTION TO CALCULATE STATS
# =========================================================

def calculate_stats(data, variable):
    
    return {
        "Mean": data[variable].mean(),
        "Median": data[variable].median(),
        "SD": data[variable].std(),
        "N": data[variable].count()
    }

# =========================================================
# TOTAL SAMPLE RESULTS
# =========================================================

total_results = []

for q in risk_questions:
    
    stats = calculate_stats(df, q)
    
    total_results.append({
        "Group": "Total Sample",
        "Question": q,
        **stats
    })

total_results_df = pd.DataFrame(total_results)

# =========================================================
# CLUSTER RESULTS
# =========================================================

cluster_results = []

for cluster in sorted(df["Cluster"].unique()):
    
    cluster_df = df[df["Cluster"] == cluster]
    
    for q in risk_questions:
        
        stats = calculate_stats(cluster_df, q)
        
        cluster_results.append({
            "Group": f"Cluster {cluster}",
            "Question": q,
            **stats
        })

cluster_results_df = pd.DataFrame(cluster_results)

# =========================================================
# COMBINE RESULTS
# =========================================================

final_results = pd.concat(
    [total_results_df, cluster_results_df],
    ignore_index=True
)

# =========================================================
# ROUND VALUES
# =========================================================

final_results[["Mean", "Median", "SD"]] = (
    final_results[["Mean", "Median", "SD"]].round(3)
)

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n================ RISK SCORE STATISTICS =================\n")
print(final_results)