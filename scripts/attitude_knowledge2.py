from scipy.stats import spearmanr
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("/Users/bazam/dev/Italian_analysis/data/datacombined-italy2.csv")

def compute_spearman(df, var1, var2):
    sub = df[[var1, var2]].dropna()
    
    if len(sub) < 3:
        return np.nan, np.nan, len(sub)
    
    rho, pval = spearmanr(sub[var1], sub[var2])
    return rho, pval, len(sub)

print("\nOverall Spearman correlations")

pairs = [
    ("knowledge_score", "attitude_score"),
    ("practice_score", "knowledge_score"),
    ("practice_score", "attitude_score")
]

for var1, var2 in pairs:
    rho, pval, n = compute_spearman(df, var1, var2)
    print(f"{var1} vs {var2}: rho={rho:.3f}, p={pval:.6f}, N={n}")

print("\nSpearman correlations per Cluster")

results = []

for cluster_id, subset in df.groupby("Cluster"):
    for var1, var2 in pairs:
        rho, pval, n = compute_spearman(subset, var1, var2)
        
        results.append({
            "Cluster": cluster_id,
            "Variable_1": var1,
            "Variable_2": var2,
            "Spearman_rho": rho,
            "p_value": pval,
            "N": n
        })
        
        print(f"Cluster {cluster_id} | {var1} vs {var2}: rho={rho:.3f}, p={pval:.4f}, N={n}")

results_df = pd.DataFrame(results)
print("\nFull results table:")
print(results_df)